from src.lib import *
from src.encoder_decoder import *

class Transformer(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,device,bias=False,lr=2.5e-4):
        super(Transformer,self).__init__()
        self.encoder = Encoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=bias)
        self.decoder = Decoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=bias)
        self.apply(self._init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.device = device

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def target_mask(self,target):
        batch_size,target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len,target_len))).expand(batch_size,1,target_len,target_len)
        return target_mask.to(self.device)
    
    def forward(self,src,target):
        encoder_out = self.encoder(src,mask=None)
        out = self.decoder(src,encoder_out,src_mask=None,target_mask=self.target_mask(target))
        #out shape: (batch_size,window_size,vocab_size)
        B,W,V = out.shape
        return out.view(B*W,V)

    def data_loader(self,data,batch_size,block_size):
        idx = torch.randint(len(data) - block_size,size=(batch_size,))
        src = torch.stack([data[i:i+block_size] for i in idx])
        target = torch.stack([data[i+1:i+1+block_size] for i in idx])

        return src.to(self.device),target.to(self.device)
    
    def fit(self,data,batch_size,block_size,n_iter):
        for _ in range(n_iter):
            src,target = self.data_loader(data,batch_size,block_size)
            logits = self.forward(src,target)
            target = target.view(logits.shape[0])
            loss   = nn.CrossEntropyLoss()(logits,target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                print(f'Iter: {_} Loss: {loss.cpu().item()}')




    def inference(self):
        pass


