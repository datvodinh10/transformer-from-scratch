import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads) -> None:
        super().__init__()

        self.embed_size = embed_size
        self.heads      = heads
        self.head_dim   = embed_size // heads
        self.values     = nn.Linear(embed_size,embed_size)
        self.keys       = nn.Linear(embed_size,embed_size)
        self.queries    = nn.Linear(embed_size,embed_size)
        self.fc_out     = nn.Linear(embed_size,embed_size)

    def forward(self,values,keys,queries,mask):
        N       = queries.shape[0]
        value_len, key_len, query_len = values.shape[1],keys.shape[1],queries.shape[1]
        
        values  = self.values(values)
        keys  = self.keys(keys)
        queries  = self.queries(queries)

        values  = values.reshape(N,value_len,self.heads,self.head_dim)
        keys    = keys.reshape(N,key_len,self.heads,self.head_dim)
        queries = queries.reshape(N,query_len,self.heads,self.head_dim)

        attention_weight   =  torch.einsum('nqhd,nkhd->nhqk',[queries,keys])
        # Queries shape: (N, query_len, heads, head_dim)
        # Keys shape: (N, key_len, heads, head_dim)
        # Attention weight shape: (N, heads, query_len, key_len)

        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask==0,float("-1e20"))

        alpha = torch.softmax(attention_weight / (self.embed_size)**(1/2),dim=3)

        out = torch.einsum('nhqk,nkhd->nqhd',[alpha,values]).reshape(N, query_len, self.heads * self.head_dim)
        # alpha shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out: (N, query_len, heads, head_dim)
        out = self.fc_out(out)

        return out