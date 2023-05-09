from src.transformer_block import *

class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
    
        super().__init__()
        self.embed_size           = embed_size
        self.device               = device
        self.word_embedding       = nn.Embedding(src_vocab_size,embed_size)
        self.positional_embedding = nn.Embedding(max_length,embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        N,seq_length = x.shape
        positions    = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        for layer in self.layers:
            out = layer(out,out,out,mask)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super().__init__()
        self.attention         = SelfAttention(embed_size,heads)
        self.norm              = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size,heads,dropout,forward_expansion)
        self.dropout           = nn.Dropout(dropout)

    def forward(self,x,values,keys,src_mask,target_mask):
        attention = self.attention(x,x,x,target_mask)
        queries   = self.dropout(self.norm(attention+x))
        out       = self.transformer_block(values,keys,queries,src_mask)
        return out
class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_length,embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,forward_expansion,dropout,device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out  = nn.Linear(embed_size,target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_out,src_mask,target_mask):
        N,seq_length = x.shape
        positions    = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x,encoder_out,encoder_out,src_mask,target_mask)

        out   = self.fc_out(x)

        return out
    
