from src.encoder_decoder import *

class Transformer(nn.Module):
    def __init__(self,
            src_vocab_size,
            target_vocab_size,
            src_pad_idx,
            target_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device='cuda',
            max_length=100):
        
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_src_mask(self,src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #(N,1,1,src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self,target):
        N,target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len,target_len))).expand(
            N,1,target_len,target_len
        )
        return target_mask.to(self.device)

    def forward(self,src,target):
        src_mask    = self.make_src_mask(src)
        target_mask = self.make_trg_mask(target)
        encoder_src = self.encoder(src,src_mask)
        out         = self.decoder(target,encoder_src,src_mask,target_mask)
        return out
    