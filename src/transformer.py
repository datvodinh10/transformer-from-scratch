from attention import *

class Transformer(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super().__init__()
        self.attention    = SelfAttention(embed_size,heads)
        self.norm1        = nn.LayerNorm(embed_size)
        self.norm2        = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,values,keys,queries,mask):
        attention = self.attention.forward(values,keys,queries,mask)
        x = self.dropout(self.norm1(attention * queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out