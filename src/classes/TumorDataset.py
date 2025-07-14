# region Dataset class definition
from torch.utils.data import Dataset
from torch import FloatTensor
from pandas import DataFrame
from typing import List, Optional


# region Dataset class definition
class TumorDataset(Dataset):
    def __init__(self,
            tumor_df: DataFrame,
            node_ids: List[str],
            norm_stats: dict,
            sample_ids_subset: Optional[List[str]] = None,
            # augment: bool = False
            ):
        self.node_ids = node_ids
        self.tumor_df = tumor_df.copy()   # should be safer from changes
        self.sample_ids = list(tumor_df.columns) \
            if sample_ids_subset is None         \
            else sample_ids_subset
        # self.augment = augment

        # Apply normalization or standardization based on norm_stats
        if {'min', 'max'}.issubset(norm_stats):
            self._apply_normalization(norm_stats)
        elif {'avg', 'std'}.issubset(norm_stats):
            self._apply_standardization(norm_stats)
        else:
            raise ValueError(
                "Invalid normalization or standardization parameters")


    def __len__(self):
        return len(self.sample_ids)


    def __getitem__(self, idx):
        sample_id     = self.sample_ids[idx]
        sample_values = self.tumor_df[sample_id].astype('float32').values

        # ? Uncomment if training is conclusive
        # ? but can still be improved by adding noise
        #if self.augment:
        #   noise = randn_like(torch.tensor(sample_values.values)) * 0.01
        #   sample_values += noise.numpy()

        return FloatTensor(sample_values)


    def _apply_standardization(self, stats: dict):
        avg_vals = stats['avg']
        std_vals = stats['std']

        std_vals[std_vals == 0] = 1.0

        self.tumor_df = self.tumor_df.sub(
            avg_vals, axis=0).div(std_vals, axis=0)


    def _apply_normalization(self, stats: dict):  # between [-1, +1]
        min_vals = stats['min']
        max_vals = stats['max']

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # Avoids division by zero

        self.tumor_df = 2 * (self.tumor_df - min_vals) / range_vals - 1

        # Clip extreme values # ? only need if extreme outliers become a problem
        #self.tumor_df = self.tumor_df.clip(-3, 3)
# endregion