import numpy as np, torch, torch.nn as nn

def make_windows(df, feats, target_col="y_ret", win=64, start=0, end=None):
    if end is None: end=len(df)-1
    X,y=[],[]
    vals=df[feats].values; targ=df[target_col].values
    for t in range(start+win, end):
        X.append(vals[t-win:t,:]); y.append(targ[t])
    return np.stack(X), np.array(y)

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,dil=1):
        super().__init__(); pad=(k-1)*dil
        self.net=nn.Sequential(
            nn.ConstantPad1d((pad,0),0.0), nn.Conv1d(in_ch,out_ch,kernel_size=k,dilation=dil), nn.ReLU(),
            nn.ConstantPad1d((pad,0),0.0), nn.Conv1d(out_ch,out_ch,kernel_size=k,dilation=dil), nn.ReLU()
        )
        self.down=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x): return self.net(x)+self.down(x)

class TCNReg(nn.Module):
    def __init__(self,in_feats=5,channels=[32,64,64],ks=3,dropout=0.2):
        super().__init__(); dil=[1,2,4][:len(channels)]; blocks=[]; c=in_feats
        for co,d in zip(channels,dil): blocks.append(ResidualBlock(c,co,k=ks,dil=d)); c=co
        self.tcn=nn.Sequential(*blocks)
        self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(p=dropout),nn.Linear(c,1))
    def forward(self,x): return self.head(self.tcn(x)).squeeze(1)
