from MPWAE2 import MPWAE2
from torch  import (
    device, Tensor, no_grad, 
    mean, cat, var, sum, mean, isinf, isnan
)
from torch.nn         import MSELoss
from torch.utils.data import DataLoader
#from torcheval.metrics import R2Score


class ModelEvaluator:
    def __init__(self,
            model: MPWAE2,
            loss: MSELoss,
            #metric: R2Score,
            test_loader: DataLoader,
            dev: device):
        self.model       = model
        self.loss        = loss
        #self.metric      = metric
        self.test_loader = test_loader
        self.device      = dev

        self.inputs  : Tensor = None
        self.outputs : Tensor = None
        self.latents : Tensor = None

        self._run_through_model()
        self.metrics = self._calculate_metrics()

        print(self.metrics)


    def _run_through_model(self):
        self.model.eval()

        all_inputs  = []
        all_outputs = []
        all_latents = []

        test_loss = 0.0

        with no_grad():
            for batch in self.test_loader:
                batch  = batch.to(self.device)
                latent = self.model.encode(batch)
                output = self.model.decode(latent)

                loss       = self.loss(output, batch)
                test_loss += loss.item() * batch.size(0)

                all_inputs.append(batch.cpu())
                all_outputs.append(output.cpu())
                all_latents.append(latent.cpu())

        concatenate  = lambda l: cat(l, dim=0)
        self.inputs  = concatenate(all_inputs)
        self.outputs = concatenate(all_outputs)
        self.latents = concatenate(all_latents)


    def _calculate_metrics(self) -> dict:
        # Check for NaN or infinite values
        if isnan(self.inputs).any() or isnan(self.outputs).any():
            print("Warning: NaN values detected in inputs or outputs")

        if isinf(self.inputs).any() or isinf(self.outputs).any():
            print("Warning: Infinite values detected in inputs or outputs")

        # Check for constant values (zero variance)
        if var(self.inputs) == 0:
            print("Warning: Input has zero variance")

        # mean square error
        mse = mean((self.inputs - self.outputs) ** 2, dim=1)

        # r2 score
        # measures how well the model performs compared to "just taking the mean"
        # r2 = 1 --> perfect reconstruction of original
        # r2 = 0 --> same quality of taking the average
        # r2 < 0 --> just take the average at this point

        ss_res = sum((self.inputs - self.outputs) ** 2)
        ss_tot = sum((self.inputs - mean(self.inputs)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        return {
            'MSE for every input': mse,
            'R2Score': r2_score,
        }