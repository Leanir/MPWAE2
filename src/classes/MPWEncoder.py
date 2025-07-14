from SparseLinear  import SparseLinear
from torch         import BoolTensor, Tensor, device
from torch.nn      import Module, Conv1d, Tanh
from torch.nn.init import kaiming_uniform_


class MPWEncoder(Module):
    def __init__(self,
            in_dim: int,
            hid_dim: int,
            cnv1_dim: int,
            in_hid_mask: BoolTensor,
            hid_cnv1_mask: BoolTensor,
            dev: device,
            # dropout: float = 0.1,
            # use_batch_norm: bool = True
            ):
        super().__init__()

        self.act = Tanh()

        # sparse layers
        self.in_hid   = SparseLinear(in_dim,  hid_dim,  in_hid_mask)# , dropout)
        self.hid_cnv1 = SparseLinear(hid_dim, cnv1_dim, hid_cnv1_mask)# , dropout)

        # conv layers
        self.cnv1_cnv2 = Conv1d(1, 1, 2, stride=2, device=dev)
        self.cnv2_out  = Conv1d(1, 1, 2, stride=2, device=dev)

        kaiming_uniform_(self.cnv1_cnv2.weight, nonlinearity='tanh')
        kaiming_uniform_(self.cnv2_out.weight, nonlinearity='tanh')

        # batch normalization layers # ? possible future implementation
        # if use_batch_norm:
        #    self.use_batch_norm = use_batch_norm
        #    self.bn1 = BatchNorm1d(hid_dim)
        #    self.bn2 = BatchNorm1d(cnv1_dim)


    def forward(self, x: Tensor) -> Tensor:
        x = self.in_hid(x)
        # if self.use_batch_norm:
        #    x = self.bn1(x)
        x = self.act(x)
        x = self.hid_cnv1(x)
        # if self.use_batch_norm:
        #    x = self.bn2(x)
        x = self.act(x)
        x = x.unsqueeze(1)
        x = self.act(self.cnv1_cnv2(x))
        x = self.act(self.cnv2_out(x))
        return x.squeeze(1)