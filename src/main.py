###############################################################################
#
# Main script to handle files, data, model and its training + evaluation
# 
# MetaPathWay AutoEncoder:
# - made by: Leandro Gozzo (aka Leanir on GitHub)
# - for:
#   - Università degli Studi di Catania
#   - Computer Science Bachelor (code L-31)
#   - Introduction to Data Mining course
#
# See README.md for more informations
#
###############################################################################


# region library versions
from sys        import   version   as python_version
from torch      import __version__ as torch_version
#!pip install torcheval --quiet
#from torcheval  import __version__ as torcheval_version
from pandas     import __version__ as pandas_version
from matplotlib import __version__ as matplotlib_version
from numpy      import __version__ as numpy_version

print(f"""Library versions:
    Python     : {python_version}
    PyTorch    : {torch_version}
    Pandas     : {pandas_version}
    Matplotlib : {matplotlib_version}
    Numpy      : {numpy_version}
""")#TorchEval  : {torcheval_version}
# endregion

# region mass import
#from google.colab import drive, files   # type: ignore
import random
from math              import ceil, log
from pandas            import read_csv
from torch             import cuda, device, manual_seed
from torch.nn          import MSELoss
from torch.optim       import Adam
from torch.utils.data  import DataLoader
#from torcheval.metrics import R2Score

# in-repo imporing
from utils.masks       import generate_masks_from_edges, expand_columns, transpose_mask
from utils.DebugHelper import DebugHelper

from classes.TumorDataset    import TumorDataset
from classes.MPWAE2          import MPWAE2
from classes.TrainingManager import TrainingManager
from classes.ModelEvaluator  import ModelEvaluator
# endregion

# region Parameter Tuning
common_seed     = 81001060  # for all pseudo-randomness purposes
s_learning_rate = 1e-3      # for Adam optimizer (lr parameter)
s_weight_decay  = 1e-5      # for Adam optimizer (L2 regularization)
s_batch_size    = 8         # for DataLoader (batch_size parameter)
s_epochs        = 100       # how many epochs for training

# strategy selector
s_patience  = ceil(log(s_epochs, 2))  # how much patience before early stopping
norm_method = "standardization"         # "normalization" | "standardization"
# ^ this currently doesn't work for normalization
#   there are still a few bugs I have not found yet
#   probable cause, wrong handling of division by zero cases where min == max

################################ PREPROCESSING ################################

# region GPU device
main_device = device("cuda:0" if cuda.is_available() else "cpu")
print(f"Running tensor operations on: {main_device}")
# endregion

# region pseudo-randomness handle
random.seed(common_seed)

manual_seed(common_seed)
if cuda.is_available():
    cuda.manual_seed_all(common_seed)
# endregion

# region data locations
#drive.mount('/content/drive')
#project_root = '/content/drive/MyDrive/AutoEncoderData'
project_root = '../data'

# TODO: assert file existence to proceed

saved_model_path = f"{project_root}/metapathway_autoencoder.pt"

tsv_root     = f'{project_root}/tsv'
nodes_path   = f"{tsv_root}/metapathway_nodes_2025.tsv"
edges_path   = f"{tsv_root}/metapathway_edges_simplified_2025.tsv"
mapping_path = f"{tsv_root}/metapathway_nodes_to_pathways_2025.tsv"
tumor_path   = f"{tsv_root}/test_tumor_samples.tsv"
# endregion

################################ READ & FILTER ################################

# region data frames
nodes_dtype   = {'#Id': 'string'}
edges_dtype   = {'#Source': 'string', 'Target': 'string'}
mapping_dtype = {'#PathwayId': 'string', 'NodeId': 'string'}
tumor_dtype   = {'NODE': 'string'}

def read_tsv(p, d):
    return read_csv(p, sep="\t", dtype=d)

nodes_df   = read_tsv(nodes_path,   nodes_dtype)
edges_df   = read_tsv(edges_path,   edges_dtype)
mapping_df = read_tsv(mapping_path, mapping_dtype)
tumor_df   = read_tsv(tumor_path,   tumor_dtype)
# endregion

# region filtering
print("Sizes before filtering:")
print(f"\tnodes_df  : {nodes_df.shape}")
print(f"\tedges_df  : {edges_df.shape}")
print(f"\tmapping_df: {mapping_df.shape}")
print(f"\ttumor_df  : {tumor_df.shape}\n")

# removing unnecessary columns
nodes_df.drop(columns=['Name', 'Type', 'Aliases'], inplace=True)
mapping_df.drop(columns=['PathwayName'], inplace=True)

# dataset filtering
all_nodes_set = set(nodes_df['#Id'])
tumor_df      = tumor_df[tumor_df['NODE'].isin(all_nodes_set)]

# node-to-node connection filtering
source_nodes_set = set(tumor_df['NODE'])      # input nodes
target_nodes_set = set(mapping_df['NodeId'])  # second layer nodes

edges_df = edges_df[edges_df['#Source'].isin(source_nodes_set)]
edges_df = edges_df[edges_df['Target'].isin(target_nodes_set)]

# not necessary but just to see how many nodes we are using from total 20507
node_percent = len(source_nodes_set | target_nodes_set) / len(all_nodes_set)

print("Sizes after filtering:")
print(f"\tnodes_df  : {nodes_df.shape}, used: {(node_percent*100):.2f}%")
print(f"\tedges_df  : {edges_df.shape}")
print(f"\tmapping_df: {mapping_df.shape}")
print(f"\ttumor_df  : {tumor_df.shape}\n")
# endregion

# region rearranging
# sorting dataframes
nodes_df.sort_values(by='#Id', inplace=True)
edges_df.sort_values(by=['#Source', 'Target'], inplace=True)
mapping_df.sort_values(by='#PathwayId', inplace=True)

# indexing of tumor_df
tumor_df.set_index('NODE', inplace=True)  # ! important

# sorted id lists of nodes and pathways
source_nodes_id_list = sorted(source_nodes_set)
target_nodes_id_list = sorted(target_nodes_set)
pathways_list        = sorted(set(mapping_df['#PathwayId']))
# endregion

#################################### MASKS ####################################

# region Mask creation
in_hid_mask = generate_masks_from_edges(
    source_nodes_id_list,
    target_nodes_id_list,
    edges_df,
    '#Source',
    'Target'
)

temp_hid_emb_temp = generate_masks_from_edges(
    target_nodes_id_list,
    pathways_list,
    mapping_df,
    "NodeId",
    "#PathwayId"
)# ^^^ this mask is expanded for hid -> conv

hid_cnv_mask = expand_columns(temp_hid_emb_temp)

# transposed masks
dcnv_hid_mask = transpose_mask(hid_cnv_mask)
hid_out_mask  = transpose_mask(in_hid_mask)

# t-upling
all_masks = (
    in_hid_mask,
    hid_cnv_mask,
    dcnv_hid_mask,
    hid_out_mask
)

print(f"Connectivity (how many active connections out of possible combos):")
print(f"\t1st mask: {in_hid_mask.sum().item()}/{in_hid_mask.numel()}")
print(f"\t2nd mask: {hid_cnv_mask.sum().item()}/{hid_cnv_mask.numel()}")
print(f"\t3st mask: {dcnv_hid_mask.sum().item()}/{dcnv_hid_mask.numel()}")
print(f"\t4th mask: {hid_out_mask.sum().item()}/{hid_out_mask.numel()}")
# endregion

################################ MODEL & STUFF ################################

# region Model instantiation
MPWAE2_model = MPWAE2(
    len(tumor_df.index),
    len(target_nodes_set),
    4*len(pathways_list),
    all_masks,
    main_device
).to(main_device)
# endregion

# region Loss instantiation
loss_func = MSELoss()
# endregion

# region Optimizer instantiation
optimizer = Adam(
    MPWAE2_model.parameters(),
    lr=s_learning_rate,
    weight_decay=s_weight_decay,
)
# endregion

# region Metric Score
#metric = R2Score() # deprecated
# endregion

############################## DATA SET & LOADER ##############################

# region Data Splitting
samples = list(tumor_df.columns)
random.shuffle(samples)

# * As there are 1110 sample columns in the original file, this code opts for an
# * approximate 70% of training set, setting it at 800 as it also is a multiple
# * of 32, which will be set as a batch size for the training cycle.
num_samples = len(samples)
num_train   = 800
remaining   = num_samples - num_train
num_val     = remaining // 2
num_test    = remaining - num_val

train_sample_ids = samples[: num_train]
val_sample_ids   = samples[num_train: num_train + num_val]
test_sample_ids  = samples[num_train + num_val:]

print(f"Total samples:      {num_samples}")
print(f"Training samples:   {len(train_sample_ids)}")
print(f"Validation samples: {len(val_sample_ids)}")
print(f"Test samples:       {len(test_sample_ids)}\n")
# endregion

# region normalization / standardization
train_df    = tumor_df[train_sample_ids]
train_stats = dict()

match norm_method:   # ? See Quick Parameter Tuning
    case "normalization":  # ! NaN errors detected
        # TODO: check for division by 0
        train_stats['min'] = train_df.min(axis=1)
        train_stats['max'] = train_df.max(axis=1)

        zero_range = train_stats['min'] == train_stats['max']

        train_stats['min'][zero_range] -= 1
        train_stats['max'][zero_range] += 1

    case "standardization":
        train_stats['avg'] = train_df.mean(axis=1)
        train_stats['std'] = train_df.std(axis=1)

        zero_std = train_stats['std'] == 0
        train_stats['std'][zero_std] = 1.0

    case _:
        raise ValueError(f"Invalid normalization method: {norm_method}")
# endregion

# region dataset creation
def create_dataset(s_ids):
    return TumorDataset(tumor_df, source_nodes_id_list, train_stats, s_ids)

train_dataset = create_dataset(train_sample_ids)
val_dataset   = create_dataset(val_sample_ids)
test_dataset  = create_dataset(test_sample_ids)
# endregion

# region data loaders
def create_dataloader(dataset: TumorDataset, batch_s: int, random: bool):
    return DataLoader(dataset, batch_size=batch_s, shuffle=random)

train_dataload = create_dataloader(train_dataset, s_batch_size, True)
val_dataload   = create_dataloader(val_dataset,   s_batch_size, False)
test_dataload  = create_dataloader(test_dataset,  s_batch_size, False)
# endregion

############################### T R A I N I N G ###############################

# region effective trainer object
trainer = TrainingManager(
    MPWAE2_model,
    optimizer,
    loss_func,
    train_dataload,
    val_dataload,
    main_device
)

loss_data = trainer.train(s_epochs, s_patience, saved_model_path)
trainer.plot_training()
# endregion

############################## FINAL  EVALUATION ##############################

# region Effective evaluator obj
evaluator = ModelEvaluator(
    MPWAE2_model,
    loss_func,
    #metric,   # deprecated use of metric object
    test_dataload,
    main_device
)
# endregion

################################# DEBUG UTILS #################################

# region actual debugger
debugger = DebugHelper(MPWAE2_model, optimizer, loss_func, main_device)

debugger.analyze_gradient_flow()
debugger.count_active_connections()
debugger.debug_data_pipeline(tumor_df, train_sample_ids)
debugger.debug_model_architecture()
debugger.debug_nan_issues(train_dataload)
#debugger.monitor_sparse_training()
debugger.test_gradient_computation(train_dataload)
# endregion
