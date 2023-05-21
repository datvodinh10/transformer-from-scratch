from src.lib import *
from src.encoder_decoder import *

class Transformer(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=False):
        self.encoder = Encoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=False)
        self.decoder = Decoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,src,target):
        encoder_out = self.encoder(src,mask=None)
        out = self.decoder(src,encoder_out,src_mask=None,target_mask=None)

