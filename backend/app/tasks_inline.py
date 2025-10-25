import os, csv, datetime as dt, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader, Dataset
from .config import SYMBOLS, START_DATE, TRAIN_START, TRAIN_END, DATA_DIR
from .data import download_prices, build_features
from .model import make_windows, TCNReg

class WindowDS(Dataset):
    def __init__(self,X,y): import torch as T; self.X=T.tensor(X,dtype=T.float32); self.y=T.tensor(y,dtype=T.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i].permute(1,0), self.y[i]

def train_epoch(model, dl, opt, device):
    model.train(); crit=torch.nn.HuberLoss(delta=1e-3); tot=n=0
    for xb,yb in dl:
        xb,yb=xb.to(device),yb.to(device)
        opt.zero_grad(); pred=model(xb); loss=crit(pred,yb); loss.backward(); opt.step()
        tot+=float(loss.item())*len(xb); n+=len(xb)
    return tot/max(n,1)

def signal_from_pred(pred_ret, vol_fore, scale_k=1.2, cap=0.6, min_vol=1e-4):
    vol=max(vol_fore, min_vol); x=pred_ret/(scale_k*vol)
    w=float(np.clip(np.tanh(x), -cap, cap))
    side="HOLD"
    if w>=0.10: side="BUY"
    elif w<=-0.10: side="SELL"
    return w, side

def run_daily():
    os.makedirs(DATA_DIR, exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    now=dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"
    rows=[]; errors=[]
    for sym in SYMBOLS:
        try:
            df_raw=download_prices(sym, START_DATE)
            if df_raw is None or len(df_raw)==0:
                errors.append(f"{sym}: no data")
                continue
            df,feats=build_features(df_raw)
            idx=df.index
            if not (idx.min()<=pd.Timestamp(TRAIN_START) and idx.max()>=pd.Timestamp(TRAIN_END)):
                errors.append(f"{sym}: insufficient history ({idx.min().date()}..{idx.max().date()})")
                continue
            train_mask=(idx>=pd.Timestamp(TRAIN_START))&(idx<pd.Timestamp(TRAIN_END))
            test_mask=(idx>=pd.Timestamp(TRAIN_END))
            train_idx=np.where(train_mask)[0]; test_idx=np.where(test_mask)[0]
            if len(train_idx)==0 or len(test_idx)==0:
                errors.append(f"{sym}: empty split")
                continue
            train_start,train_end=train_idx[0],train_idx[-1]
            test_start=test_idx[0]
            Xtr,Ytr=make_windows(df,feats,"y_ret",win=64,start=train_start,end=train_end)
            Xte,Yte=make_windows(df,feats,"y_ret",win=64,start=test_start,end=len(df)-1)
            if len(Xtr)==0 or len(Xte)==0:
                errors.append(f"{sym}: no windows")
                continue
            dl=DataLoader(WindowDS(Xtr,Ytr),batch_size=256,shuffle=True,drop_last=True)
            model=TCNReg(in_feats=len(feats)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
            for _ in range(8): _=train_epoch(model,dl,opt,device)
            import torch as T
            xb=T.tensor(Xte[-1:],dtype=T.float32).permute(0,2,1).to(device)
            with T.no_grad(): pred=float(model(xb).cpu().numpy().ravel()[0])
            vol_all=df["ret1"].ewm(span=20).std().shift(1).bfill().values
            idx0=test_start+64; vol_fore=float(vol_all[idx0+len(Xte)-1])
            w,side=signal_from_pred(pred,vol_fore,1.2,0.6)
            price=float(df["Close"].iloc[-1]); rows.append([now,sym,side,w,price])
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {e}")
    sig_path=os.path.join(DATA_DIR,"signals.csv")
    write_header=not os.path.exists(sig_path)
    if rows:
        with open(sig_path,"a",newline="") as f:
            wr=csv.writer(f)
            if write_header: wr.writerow(["ts","symbol","side","weight","price"])
            wr.writerows(rows)
    err_path=os.path.join(DATA_DIR,"last_errors.log")
    if errors:
        with open(err_path,"w") as f:
            for line in errors: f.write(line+"\n")
    else:
        if os.path.exists(err_path): os.remove(err_path)
    return rows, errors
