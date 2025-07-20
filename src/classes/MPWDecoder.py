from SparseLinear  import SparseLinear
from torch         import Tensor, BoolTensor, device, Module
from torch.nn      import ConvTranspose1d#, Tanh
from torch.nn.init import kaiming_uniform_
from ArcTanh       import ArcTanh

class MPWDecoder(Module):
    def __init__(self,
            dcnv2_dim: int,
            hid_dim: int,
            out_dim: int,
            dcnv2_hid_mask: BoolTensor,
            hid_out_mask: BoolTensor,
            dev: device,
            # dropout: float = 0.1,
            # use_batch_norm: bool = True
            ):
        super().__init__()

        self.act = ArcTanh()

        # transp conv layers
        self.emb_dcnv1   = ConvTranspose1d(1, 1, 2, stride=2, device=dev)
        self.dcnv1_dcnv2 = ConvTranspose1d(1, 1, 2, stride=2, device=dev)

        kaiming_uniform_(self.emb_dcnv1.weight, nonlinearity='tanh')
        kaiming_uniform_(self.dcnv1_dcnv2.weight, nonlinearity='tanh')

        # Sparse layers (with transposed masks)
        self.dcnv2_hid = SparseLinear(dcnv2_dim, hid_dim, dcnv2_hid_mask)# , dropout)
        self.hid_out   = SparseLinear(hid_dim,   out_dim, hid_out_mask)  # , dropout)

        # Batch normalization # ? possible future implementation
        # if use_batch_norm:
        #    self.use_batch_norm = use_batch_norm
        #    self.bn1 = nn.BatchNorm1d(dcnv2_dim)
        #    self.bn2 = nn.BatchNorm1d(hid_dim)


    def forward(self, x: Tensor) -> Tensor:
        # Transposed convolutional layers
        x = x.unsqueeze(1)
        x = self.act(self.emb_dcnv1(x))
        x = self.act(self.dcnv1_dcnv2(x))
        x = x.squeeze(1)
        x = self.dcnv2_hid(x)
        # if self.use_batch_norm:
        #    x = self.bn1(x)
        x = self.act(x)
        x = self.hid_out(x)
        # if self.use_batch_norm:
        #    x = self.bn2(x)
        return self.act(x)