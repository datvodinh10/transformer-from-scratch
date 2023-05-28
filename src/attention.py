from src.lib import *

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size,heads,bias=False):
        super(MultiHeadAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = int(embed_size / heads)
        self.keys = nn.Linear(embed_size,embed_size,bias=bias)
        self.queries = nn.Linear(embed_size,embed_size,bias=bias)
        self.values = nn.Linear(embed_size,embed_size,bias=bias)
        self.fc = nn.Linear(embed_size,embed_size,bias=bias)

    def forward(self,key,query,value,mask=None):

        # key shape: (batch_size,key_len,embed_size)
        # query shape: (batch_size,query_len,embed_size)
        # value shape: (batch_size,value_len,embed_size)
        # key and query and value all have the same shape
 

        keys = self.keys(key).reshape(key.shape[0],key.shape[1],self.heads,self.heads_dim)
        queries = self.queries(query).reshape(query.shape[0],query.shape[1],self.heads,self.heads_dim)
        values = self.values(value).reshape(value.shape[0],value.shape[1],self.heads,self.heads_dim)

        # keys shape: (batch_size,key_len,heads,head_dim)
        # queries shape: (batch_size,query_len,heads,head_dim)
        # values shape: (batch_size,value_len,heads,head_dim)

        dot_product = torch.einsum('bkhd,bqhd->bhqk',keys,queries)
        
        # dot_product shape: (batch_size,heads,query_len,key_len)
        if mask is not None:
            dot_product = dot_product.masked_fill(mask==0,float('-inf'))

        scaled_product = torch.softmax(dot_product / (self.embed_size)**(1/2),dim=3)

        alpha = torch.einsum("bhqk,bvhd->bqhd",scaled_product,values)
        out = self.fc(alpha.reshape(key.shape[0],key.shape[1],self.embed_size))

        return out

        




