from torch    import Tensor, atanh
from torch.nn import Module

class ArcTanh(Module):

    def forward(self, input: Tensor) -> Tensor:
        domain = input.min().item() > -1 and input.max().item() < 1
        assert domain, "Domain for arctanh not respected"
        return atanh(input)