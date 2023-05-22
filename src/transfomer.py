from src.lib import *
from src.encoder_decoder import *

class Transformer(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,device,bias=False,lr=2.5e-4):
        super(Transformer,self).__init__()
        self.encoder = Encoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=bias).to(device)
        self.decoder = Decoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=bias).to(device)
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
    
    def forward(self,src):
        encoder_out = self.encoder(src,mask=None)
        out = self.decoder(src,encoder_out,src_mask=None,target_mask=self.target_mask(src))
        #out shape: (batch_size,window_size,vocab_size)
        
        return out

    def data_loader(self,data,batch_size,block_size):
        idx = torch.randint(len(data) - block_size,size=(batch_size,))
        src = torch.stack([data[i:i+block_size] for i in idx])
        target = torch.stack([data[i+1:i+1+block_size] for i in idx])

        return src.to(self.device),target.to(self.device)
    
    def fit(self,data,batch_size,block_size,n_iter):
        self.block_size = block_size
        for _ in range(n_iter):
            src,target = self.data_loader(data,batch_size,block_size)
            logits = self.forward(src)
            B,W,V = logits.shape
            logits = logits.view(B*W,V)
            target = target.view(logits.shape[0])
            loss   = nn.CrossEntropyLoss()(logits,target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                print(f'Iter: {_} Loss: {loss.cpu().item()}')




    def inference(self,src,max_token = 0):
        with torch.no_grad():
            for _ in range(max_token):
                src_in = src[:,-self.block_size:]
                logits = self.forward(src_in)
                logits = logits[:,-1,:]
                probs = F.softmax(logits, dim=-1)
                src_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                src = torch.cat((src, src_next), dim=1) # (B, T+1)
            return src







