from src.lib import *

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout = 0.5,max_len=1000):
        super(PositionalEncoding,self).__init__()
        PE = torch.zeros((1,max_len,num_hiddens))
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0,max_len,dtype=torch.float32).reshape(-1,1) \
        / torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32) / num_hiddens)
        PE[:,:,0::2] = torch.sin(position)
        PE[:,:,1::2] = torch.cos(position)
        self.register_buffer('PE',PE)


    def forward(self,x):
        x = x + self.PE[:,:x.shape[1],:]
        return self.dropout(x)