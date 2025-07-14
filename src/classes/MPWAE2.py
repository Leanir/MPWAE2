from MPWEncoder import MPWEncoder
from MPWDecoder import MPWDecoder
from torch import Module, device, Tensor


class MPWAE2(Module):
    def __init__(self,
            fll_dim: int,   # first and last layers' size
            hiho_dim: int,  # first and last hidden layers' size
            cnv_dim: int,   # first [de]convolutional layer's size
            masks: tuple,   # all the masks necessary, compacted
            dev: device,
            # dropout: float = 0.1,
            # use_batch_norm: bool = True
            ):
        super().__init__()

        in_hid, hid_cnv, dcnv_hid, hid_out = masks

        self.encoder = MPWEncoder(
            fll_dim,
            hiho_dim,
            cnv_dim,
            in_hid,
            hid_cnv,
            dev
            #, dropout, use_batch_norm
        )

        self.decoder = MPWDecoder(
            cnv_dim,
            hiho_dim,
            fll_dim,
            dcnv_hid,
            hid_out,
            dev
            #, dropout, use_batch_norm
        )


    def forward(self, x: Tensor) -> Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)


    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)