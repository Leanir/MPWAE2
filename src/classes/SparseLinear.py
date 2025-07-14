from torch         import BoolTensor, Tensor, empty, zeros, functional as F
from torch.nn      import Module, Parameter
from torch.nn.init import xavier_uniform_, calculate_gain

class SparseLinear(Module):
    def __init__(self,
            in_dim: int,
            out_dim: int,
            connect_mask: BoolTensor,
            # dropout: float = 0.0
            ):
        super().__init__()

        # Xavier/Glorot initialization for better gradient flow
        w = empty((out_dim, in_dim))
        xavier_uniform_(w, gain=calculate_gain('tanh'))  # in-place

        self.weight = Parameter(w)
        self.bias = Parameter(zeros(out_dim))

        # Register mask as buffer (non-trainable)
        self.register_buffer("mask", connect_mask)

        # dropout for regularization
        # self.dropout = Dropout(dropout) if dropout > 0 else None


    def forward(self, x: Tensor) -> Tensor:
        # Apply mask to weight matrix
        masked_weight = self.weight * self.mask
        output = F.linear(x, masked_weight, self.bias)

        # if self.dropout is not None:
        #    output = self.dropout(output)

        return output
