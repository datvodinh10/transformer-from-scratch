from src.lib import *
from src.attention import *
from src.positional_encoding import *
class EncoderBlock(nn.Module):
    def __init__(self,embed_size,heads,bias=False):
        super(EncoderBlock,self).__init__()
        self.attention    = MultiHeadAttention(embed_size,heads,bias)
        self.layer_norm1  = nn.LayerNorm()
        self.layer_norm2  = nn.LayerNorm()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size,embed_size)
        )

    def forward(self,key,query,value,mask=None):
        attention = self.attention(key,query,value,mask)
        out = self.layer_norm1(key + attention)
        out_ffw = self.feed_forward(out)
        out = self.layer_norm2(out + out_ffw)

        return out

class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=False):
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.position_embed = PositionalEncoding(vocab_size,max_len=max_len)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(embed_size,heads,bias)
                for _ in range(num_layers)
            ]

        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        x_embed = self.embed(x)
        x_embed = self.position_embed(x_embed)
        out = self.dropout(x_embed)
        for layer in self.encoder_layers:
            out = layer(out,out,out,mask)
    
        return out


class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,bias=False):
        super(DecoderBlock,self).__init__()
        self.encoder_block = EncoderBlock(embed_size,heads,bias)
        self.attention = MultiHeadAttention(embed_size,heads,bias)
        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout()

    def forward(self,x,enc_value,enc_key,src_mask,target_mask):
        out = self.layer_norm(x + self.attention(x,x,x,src_mask))
        out = self.dropout(out)
        out = self.encoder_block(key=enc_key,value=enc_value,query=out,mask=target_mask)

        return out


class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=False):
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.position_embed = PositionalEncoding(embed_size,max_len=max_len)
        self.decoder_layer = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,bias)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(embed_size,vocab_size)

    def forward(self,x,encoder_out,src_mask,target_mask):
        x_embed = self.embed(x)
        x_embed = self.position_embed(x_embed)
        out = self.dropout(x_embed)
        for layer in self.decoder_layer:
            out = layer(out,encoder_out,encoder_out,src_mask,target_mask)

        out = self.fc(out)

        return out


