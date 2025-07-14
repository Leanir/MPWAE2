from pandas import DataFrame
from torch import BoolTensor, repeat_interleave, zeros


def generate_masks_from_edges(
        sources: list[str],
        targets: list[str],
        edges: DataFrame,
        source_col: str,
        target_col: str) -> BoolTensor:
    is_same_list = sources == targets

    num_src = len(sources)
    num_trg = num_src if is_same_list else len(targets)

    src_index = {v: i for i, v in enumerate(sources)}
    trg_index = src_index \
        if is_same_list   \
        else {v: i for i, v in enumerate(targets)}

    forward_mask = zeros((num_trg, num_src), dtype=bool)
    for src, trg in edges[[source_col, target_col]].itertuples(index=False):
        if src in src_index and trg in trg_index:
            forward_mask[trg_index[trg], src_index[src]] = True

    return forward_mask


def expand_columns(mask: BoolTensor, k: int = 2) -> BoolTensor:
    return repeat_interleave(mask, 2 ** k, dim=0)


def transpose_mask(mask: BoolTensor) -> BoolTensor:
    return mask.T.clone()
