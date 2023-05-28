from src.lib import *
from src.encoder_decoder import *

class TransformerModel(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,device,decode_vocab,bias=False,lr=2.5e-4):
        super(TransformerModel,self).__init__()
        self.encoder = Encoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=bias).to(device)
        self.decoder = Decoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=bias).to(device)
        self.apply(self._init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.device = device
        self.decode_vocab = decode_vocab
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
    
    def fit(self,data,batch_size,block_size,n_iter,print_every=50):
        self.loss = []
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
            self.loss.append(loss.detach().cpu().item())
            if _%print_every==0:
                with torch.no_grad():
                    # context = data[:50].reshape(1,-1).to(self.device)
                    context = torch.zeros((1,1),dtype=torch.long,device=self.device)
                    print(f'Iter: {_} Loss: {loss.cpu().item()}')
                    self.inference(context, max_token=20)
                    # print(f'Inference: {self.decode_vocab(self.inference(context, max_token=50)[0].tolist())}')
                    print('----------------------------------')

    def plot(self):
        sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=18)     # fontsize of the axes title
        plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
        plt.rc('legend', fontsize=13)    # legend fontsize
        plt.rc('font', size=13)
        plt.figure(figsize=(12,4),clear=True)
        plt.plot(self.loss)
        plt.ylabel('Loss')



    def inference(self,src,max_token = 0,infer=False):
        with torch.no_grad():
            for _ in range(max_token):
                src_in = src[:,-self.block_size:]
                logits = self.forward(src_in)
                logits = logits[:,-1,:]
                probs = F.softmax(logits, dim=-1)
                src_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # print(self.decode_vocab(list([src_next.item()])),end=" ")
                src = torch.cat((src, src_next), dim=1) # (B, T+1)
            print(self.decode_vocab(src))
            # return src







