"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Kult2012 Dataset                                                            ║
║  Methods : CTGAN │ Diffusion │ HiDe-Tab │ TabDiff │ TabSyn                  ║
║  Ablation: 1Stage │ 3Stage │ noTC │ noFiLM                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = (
    r"D:\PhD\PROJECT\OPEN SCIENCE II\New dataset\dataverse_files"
    r"\Kult2012_3kraje_UstVysZli_CSDA_pub_nove_bez_jmen.sav"
)

# Choose targer sample size: 5000 / 10000 / 20000
N_SYNTH        = 5000
PURE_SYNTHETIC = True

METHODS = [
    "CTGAN",
    "Diffusion",
    "HiDe_Tab",          # full model (two-stage, TC, FiLM)
    "HiDeTab_noTC",      # ablation: no TC penalty
    "HiDeTab_1Stage",    # ablation: one flat diffusion stage
    "HiDeTab_3Stage",    # ablation: three sequential stages
    "HiDeTab_noFiLM",    # ablation: no FiLM conditioning in Stage B
    "TabDiff",
    "TabSyn",
]

# ── epochs ────────────────────────────────────────────────────────────────────
EPOCHS_VAE          = 300   # CVAE / T-CVAE standalone
EPOCHS_VAE_PRETRAIN = 120   # VAE inside latent-diffusion methods
EPOCHS_GAN          = 200
EPOCHS_DIFF         = 300   # diffusion denoiser epochs
EPOCHS_FINETUNE     = 80    # joint fine-tune

# ── architecture ──────────────────────────────────────────────────────────────
BATCH_SIZE         = 128
LR                 = 1e-4
LATENT_DIM         = 64
HIDDEN_DIM         = 512
TRANSFORMER_STRIDE = 16
N_HEADS            = 4
N_LAYERS           = 3
T_STEPS            = 300
N_CLUSTERS         = 16

# ── loss weights ─────────────────────────────────────────────────────────────
NUM_LOSS_WEIGHT  = 80.0     # upweight numeric in VAE reconstruction
NUM_FT_EXTRA     = 6.0      # extra ×N on numeric during joint FT decoder pass
CTGAN_NUM_WEIGHT = 5.0      # auxiliary numeric loss weight in CTGAN generator

# ── sampling ─────────────────────────────────────────────────────────────────
CAT_TEMP  = 0.8
EMA_DECAY = 0.9999

# ── preprocessing ─────────────────────────────────────────────────────────────
MAX_CAT_LEVELS = 15
MAX_MISS_FRAC  = 0.50
MAX_CAT_UNIQUE = 100

OUTPUT_DIR = "outputs_kult2012"

# ══════════════════════════════════════════════════════════════════════════════

import os, sys, time, math, warnings, copy
warnings.filterwarnings("ignore")
import random
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.exit("[ERROR] PyTorch not found.")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    _HAS_MANIFOLD = True
except ImportError:
    _HAS_MANIFOLD = False

# ── device ────────────────────────────────────────────────────────────────────
def _select_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        nm = torch.cuda.get_device_name(0)
        mb = torch.cuda.get_device_properties(0).total_memory/1024**3
        print(f"\n{'='*70}\n[GPU] {nm}  {mb:.2f} GB\n{'='*70}\n")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return d
    print("[GPU] CPU only.")
    return torch.device("cpu")

DEVICE = _select_device()
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda": torch.cuda.manual_seed_all(SEED)

OUT = Path(OUTPUT_DIR); PLOT_DIR = OUT/"plots"
OUT.mkdir(exist_ok=True); PLOT_DIR.mkdir(exist_ok=True)

try:
    import pyreadstat; _HAS_PYR = True
except ImportError:
    _HAS_PYR = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon
from scipy import stats as scipy_stats

# ==============================================================================
# 1.  CZECH MISSING-VALUE TOKENS (unchanged)
# ==============================================================================
_MV = frozenset({"BEZ ODPOVĚDI","BEZ ODPOVEDI","BEZ ODPOV","BEZ ODPOV.",
                  "NEV.","NEV","NEVÍM","NEVIM","NEV  ",
                  "NIC NENAVTÍVIL","NIC NENAVTIVIL"})

def _is_missing(v):
    if v is None: return True
    if isinstance(v, float) and math.isnan(v): return True
    if isinstance(v, str):
        s = v.strip().upper()
        if s in _MV: return True
        for t in _MV:
            if s.startswith(t): return True
    return False

def _to_plain(s):
    return s.astype(object) if hasattr(s,"cat") else s

def _clean(s):
    return _to_plain(s).apply(lambda v: np.nan if _is_missing(v) else v)

# ==============================================================================
# 2.  DATA LOADING (unchanged)
# ==============================================================================
_ENCS = ["cp1250","iso-8859-2","latin-1","utf-8","cp1252"]

def load_sav(path):
    if _HAS_PYR:
        for enc in _ENCS:
            try:
                df,_ = pyreadstat.read_sav(path, apply_value_formats=True, encoding=enc)
                print(f"[load] pyreadstat OK ({enc}) → {df.shape}"); return df
            except Exception as e: print(f"[load] {enc}: {e}")
    try:
        df = pd.read_spss(path); print(f"[load] read_spss → {df.shape}"); return df
    except Exception as e: raise RuntimeError(e) from e

# ==============================================================================
# 3.  DATA PROCESSOR (unchanged)
# ==============================================================================

class ColumnSpec:
    __slots__ = ("name","kind","is_int","fill","scaler","cats","fill_cat","n")
    def __init__(self, name):
        self.name=name; self.kind=None; self.is_int=False
        self.fill=None; self.scaler=MinMaxScaler(clip=True)
        self.cats=None; self.fill_cat=None; self.n=1

    def fit(self, raw):
        c = _clean(raw)
        if pd.api.types.is_numeric_dtype(c):
            self.kind = "num"
            num = pd.to_numeric(c, errors="coerce")
            valid = num.dropna()
            self.is_int = len(valid)>0 and bool((valid==valid.round()).all())
            self.fill = float(valid.median()) if len(valid)>0 else 0.0
            self.scaler.fit(num.fillna(self.fill).values.reshape(-1,1))
            self.n = 1
        else:
            self.kind = "cat"
            s = c.astype(str).where(c.notna(), other=np.nan)
            vc = s.dropna().value_counts()
            top = list(vc.index[:MAX_CAT_LEVELS])
            self.fill_cat = top[0] if top else "__UNK__"
            self.cats = top; self.n = len(top)
        return self

    def transform(self, raw):
        c = _clean(raw)
        if self.kind == "num":
            v = pd.to_numeric(c, errors="coerce").fillna(self.fill)
            return self.scaler.transform(v.values.reshape(-1,1)).ravel().astype(np.float32)
        s = c.astype(str).where(c.notna(),np.nan).fillna(self.fill_cat).values
        K = len(self.cats); mat = np.zeros((len(s),K),np.float32)
        ci = {c:i for i,c in enumerate(self.cats)}
        for r,v in enumerate(s): mat[r, ci.get(v, ci.get(self.fill_cat,0))] = 1.0
        return mat

    def inverse_transform(self, arr):
        if self.kind == "num":
            u = self.scaler.inverse_transform(np.asarray(arr).reshape(-1,1)).ravel()
            return np.round(u).astype(np.int64) if self.is_int else u
        mat = np.asarray(arr)
        if mat.ndim==1: mat=mat.reshape(-1,1)
        idx = np.clip(mat.argmax(1), 0, len(self.cats)-1)
        return np.array(self.cats)[idx]


class DataProcessor:
    def __init__(self):
        self.columns=[]; self.specs=[]; self.slices=[]; self._n=0

    def fit(self, df):
        self.columns = list(df.columns)
        self.specs   = [ColumnSpec(c).fit(df[c]) for c in self.columns]
        i = 0
        for sp in self.specs: self.slices.append((i, i+sp.n)); i+=sp.n
        self._n = i
        nn_ = sum(s.kind=="num" for s in self.specs)
        nc_ = sum(s.kind=="cat" for s in self.specs)
        print(f"[proc] {len(self.columns)} cols → {i} dims  (num={nn_}, cat={nc_})")
        return self

    def transform(self, df):
        parts=[]
        for sp,c in zip(self.specs,self.columns):
            o=sp.transform(df[c]); parts.append(o.reshape(-1,1) if o.ndim==1 else o)
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df): return self.fit(df).transform(df)

    def inverse_transform(self, X):
        out={}
        for sp,(s,e) in zip(self.specs,self.slices):
            chunk=X[:,s:e]
            out[sp.name]=sp.inverse_transform(chunk.ravel() if sp.kind=="num" else chunk)
        return pd.DataFrame(out)

    @property
    def n_features(self): return self._n
    @property
    def num_dims(self): return [s for sp,(s,e) in zip(self.specs,self.slices) if sp.kind=="num"]
    @property
    def cat_groups(self): return [(s,e,e-s) for sp,(s,e) in zip(self.specs,self.slices) if sp.kind=="cat"]
    # Map flat encoded index → original column name
    def col_name(self, enc_dim):
        for sp,(s,e) in zip(self.specs,self.slices):
            if s<=enc_dim<e: return sp.name
        return f"dim_{enc_dim}"

# ==============================================================================
# 4.  PREPROCESSING (unchanged)
# ==============================================================================

def preprocess_df(df):
    n0 = df.shape[1]
    dm = list(df.isnull().mean()[df.isnull().mean()>MAX_MISS_FRAC].index)
    cat_c = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    dh = [c for c in cat_c if df[c].nunique()>MAX_CAT_UNIQUE]
    di = [c for c in ["CD"] if c in df.columns]
    drop = set(dm+dh+di)
    df2 = df.drop(columns=[c for c in drop if c in df.columns])
    for c in df2.columns:
        if not pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = _to_plain(df2[c]).apply(lambda v: np.nan if _is_missing(v) else v)
    print(f"[prep] dropped {len(drop)} ({len(dm)} miss, {len(dh)} hc, {len(di)} id)")
    print(f"[prep] {n0} → {df2.shape[1]} cols  ({df2.shape[0]} rows)")
    return df2

# ==============================================================================
# 5.  HELPERS (unchanged)
# ==============================================================================

def make_loader(X, labels=None, batch=None, shuffle=True):
    b = batch or BATCH_SIZE
    ds = TensorDataset(X) if labels is None else TensorDataset(X, labels)
    pin = (DEVICE.type=="cuda") and (not X.is_cuda)
    return DataLoader(ds, batch_size=b, shuffle=shuffle, pin_memory=pin, num_workers=0)

def gpu(t): return t.to(DEVICE, non_blocking=True)

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k:v.cpu().clone().float() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            self.shadow[k] = self.decay*self.shadow[k] + (1-self.decay)*v.cpu().float()
    def apply(self, model):
        model.load_state_dict({k:v.to(DEVICE) for k,v in self.shadow.items()})

# ==============================================================================
# 6.  LOSS (unchanged)
# ==============================================================================

def composite_loss(recon, target, cat_groups, num_dims,
                   beta_ce=1.0, num_weight=NUM_LOSS_WEIGHT):
    loss = torch.tensor(0.0, device=recon.device); n=0
    for s,e,K in cat_groups:
        loss += beta_ce * F.cross_entropy(recon[:,s:e],
                                           target[:,s:e].argmax(1),
                                           label_smoothing=0.05)
        n += 1
    if num_dims:
        idx = torch.tensor(num_dims, device=recon.device, dtype=torch.long)
        loss += num_weight * F.mse_loss(recon[:,idx], target[:,idx])
        n += 1
    return loss / max(n,1)

def vae_loss(recon, target, mu, lv, cat_groups, num_dims, beta_kl=0.1):
    return (composite_loss(recon, target, cat_groups, num_dims)
            - 0.5*beta_kl*torch.mean(1+lv-mu.pow(2)-lv.exp()))

def _cyc_beta(ep, epochs, cycles=6, bmax=0.05):
    cl = max(epochs//cycles, 1)
    return bmax * min((ep%cl)/cl*2.0, 1.0)

# ==============================================================================
# 7.  MULTI-HEAD DECODER  (separate numeric trunk) – unchanged
# ==============================================================================

class MultiHeadDecoder(nn.Module):
    def __init__(self, lat, hid, cat_groups, num_dims, total_dims):
        super().__init__()
        self.cat_groups=cat_groups; self.num_dims=num_dims; self.total=total_dims
        # Cat trunk
        self.trunk = nn.Sequential(
            nn.Linear(lat,hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid,hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid,hid), nn.LayerNorm(hid), nn.GELU())
        self.cat_heads = nn.ModuleList([nn.Linear(hid,K) for _,_,K in cat_groups])
        # Dedicated numeric trunk (no gradient interference from cat loss)
        if num_dims:
            nh = max(hid//2, 64)
            self.num_trunk = nn.Sequential(
                nn.Linear(lat,nh), nn.LayerNorm(nh), nn.GELU(),
                nn.Linear(nh,nh),  nn.LayerNorm(nh), nn.GELU(),
                nn.Linear(nh,nh),  nn.GELU())
            self.num_head = nn.Linear(nh, len(num_dims))
        else:
            self.num_trunk = self.num_head = None

    def forward(self, z):
        h = self.trunk(z)
        out = torch.zeros(z.size(0), self.total, device=z.device)
        for head,(s,e,K) in zip(self.cat_heads, self.cat_groups): out[:,s:e]=head(h)
        if self.num_head is not None:
            idx = torch.tensor(self.num_dims, device=z.device, dtype=torch.long)
            out[:,idx] = torch.sigmoid(self.num_head(self.num_trunk(z)))
        return out

    @torch.no_grad()
    def sample(self, z, temperature=CAT_TEMP):
        h = self.trunk(z)
        out = torch.zeros(z.size(0), self.total, device=z.device)
        for head,(s,e,K) in zip(self.cat_heads, self.cat_groups):
            probs = F.softmax(head(h)/max(temperature,1e-6), dim=-1)
            oh = F.one_hot(torch.multinomial(probs,1).squeeze(-1),K).float()
            out[:,s:e] = oh
        if self.num_head is not None:
            idx = torch.tensor(self.num_dims, device=z.device, dtype=torch.long)
            out[:,idx] = torch.sigmoid(self.num_head(self.num_trunk(z)))
        return out

# ==============================================================================
# 8.  ENCODER (unchanged)
# ==============================================================================

class _Res(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(0.1),nn.Linear(d*2,d))
    def forward(self, x): return x+self.ff(self.ln(x))

class CVAEEnc(nn.Module):
    def __init__(self, in_d, hid, lat):
        super().__init__()
        mid=min(hid*2,1024)
        self.proj = nn.Sequential(nn.Linear(in_d,mid),nn.LayerNorm(mid),nn.GELU(),
                                   nn.Linear(mid,hid), nn.LayerNorm(hid), nn.GELU())
        self.res  = nn.Sequential(*[_Res(hid) for _ in range(6)])
        self.norm = nn.LayerNorm(hid)
        self.mu   = nn.Linear(hid,lat)
        self.lv   = nn.Linear(hid,lat)
    def forward(self, x):
        h=self.norm(self.res(self.proj(x)))
        return self.mu(h), self.lv(h).clamp(-4,4)

# ==============================================================================
# 9.  CVAE MODEL + TRAINING (unchanged)
# ==============================================================================

class CVAE(nn.Module):
    def __init__(self, in_d, cat_groups, num_dims, hid=HIDDEN_DIM, lat=LATENT_DIM):
        super().__init__()
        self.enc = CVAEEnc(in_d, hid, lat)
        self.dec = MultiHeadDecoder(lat, hid, cat_groups, num_dims, in_d)
        self.lat = lat
    def _rp(self, mu, lv):
        return mu + torch.exp(0.5*lv.clamp(-10,10))*torch.randn_like(mu)
    def forward(self, x):
        mu,lv=self.enc(x); return self.dec(self._rp(mu,lv)), mu, lv
    @torch.no_grad()
    def sample(self, n):
        self.eval()
        return self.dec.sample(torch.randn(n,self.lat,device=DEVICE)).cpu().numpy()
    @torch.no_grad()
    def encode_mu(self, x): return self.enc(x)[0]


def _train_vae_core(model, X, proc, epochs, cyc_kl=False, tag="VAE",
                     loss_history=None):
    """Shared training loop for all VAE variants."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR*3 if cyc_kl else LR,
                             weight_decay=1e-4 if cyc_kl else 1e-5)
    if cyc_kl:
        T0=max(epochs//6,10)
        sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T0,eta_min=LR*0.05)
    else:
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    ldr = make_loader(X)
    best, bst = float("inf"), None
    for ep in range(1, epochs+1):
        model.train(); tot=0.0
        bkl = _cyc_beta(ep,epochs) if cyc_kl else min(0.1, ep/epochs*0.1)
        for (xb,) in ldr:
            xb=gpu(xb); r,mu,lv=model(xb)
            # Track reconstruction alone for checkpointing when using cyclical KL
            rl=composite_loss(r,xb,proc.cat_groups,proc.num_dims)
            kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            loss=rl+bkl*kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5 if not cyc_kl else 1.0)
            opt.step(); tot+=rl.item()
        sch.step(); ep_l=tot/len(ldr)
        if ep_l<best: best=ep_l; bst={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if loss_history is not None: loss_history.append(ep_l)
        if ep%50==0 or ep==epochs:
            print(f"    [{tag}]  ep {ep:4d}/{epochs}  recon={ep_l:.5f}  β={bkl:.4f}")
    model.load_state_dict({k:v.to(DEVICE) for k,v in bst.items()})
    return model


# ==============================================================================
# 11.  CTGAN (unchanged)
# ==============================================================================

class _CTGANGen(nn.Module):
    def __init__(self, nd, hid, cat_groups, num_dims, total_dims):
        super().__init__()
        self.fc0=nn.Sequential(nn.Linear(nd,hid),  nn.BatchNorm1d(hid), nn.ReLU())
        self.fc1=nn.Sequential(nn.Linear(hid,hid), nn.BatchNorm1d(hid), nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(hid,hid), nn.BatchNorm1d(hid), nn.ReLU())
        self.fc3=nn.Sequential(nn.Linear(hid,hid), nn.BatchNorm1d(hid), nn.ReLU())
        self.skip=nn.Linear(nd,hid)
        self.head=MultiHeadDecoder(hid,hid,cat_groups,num_dims,total_dims)
    def _trunk(self,z):
        h0=self.fc0(z); h1=self.fc1(h0); h2=self.fc2(h1+h0)
        return self.fc3(h2)+self.skip(z)
    def forward(self,z): return self.head(self._trunk(z))
    def sample(self,z):  return self.head.sample(self._trunk(z))

class _CTGANDisc(nn.Module):
    def __init__(self, id_, hid):
        super().__init__()
        SN=nn.utils.spectral_norm
        self.net=nn.Sequential(
            SN(nn.Linear(id_,hid)),     nn.LeakyReLU(0.2),nn.Dropout(0.25),
            SN(nn.Linear(hid,hid)),     nn.LeakyReLU(0.2),nn.Dropout(0.25),
            SN(nn.Linear(hid,hid//2)),  nn.LeakyReLU(0.2),
            SN(nn.Linear(hid//2,1)))
    def forward(self,x): return self.net(x)

def _gp(D, real, fake, lam=10.0):
    a=torch.rand(real.size(0),1,device=real.device)
    ip=(a*real+(1-a)*fake).requires_grad_(True)
    g=torch.autograd.grad(D(ip),ip,
        grad_outputs=torch.ones(ip.size(0),1,device=ip.device),
        create_graph=True)[0]
    return lam*((g.norm(2,dim=1)-1)**2).mean()

def train_ctgan(X, proc, epochs=EPOCHS_GAN, nd=LATENT_DIM, loss_history=None):
    id_=X.shape[1]
    G=_CTGANGen(nd,HIDDEN_DIM,proc.cat_groups,proc.num_dims,id_).to(DEVICE)
    D=_CTGANDisc(id_,HIDDEN_DIM).to(DEVICE)
    oG=torch.optim.Adam(G.parameters(),lr=LR*0.5,betas=(0.5,0.9))
    oD=torch.optim.Adam(D.parameters(),lr=LR,    betas=(0.5,0.9))
    sG=torch.optim.lr_scheduler.CosineAnnealingLR(oG,epochs)
    sD=torch.optim.lr_scheduler.CosineAnnealingLR(oD,epochs)
    ldr=make_loader(X)

    for ep in range(1,epochs+1):
        dL=gL=0.0
        for (xb,) in ldr:
            xb=gpu(xb); bs=xb.size(0)
            # 5 D-steps
            for _ in range(5):
                z=torch.randn(bs,nd,device=DEVICE); fk=G(z).detach()
                ld=-D(xb).mean()+D(fk).mean()+_gp(D,xb,fk)
                oD.zero_grad(); ld.backward(); oD.step()
            # 2 G-steps with numeric auxiliary loss
            for _ in range(2):
                z=torch.randn(bs,nd,device=DEVICE); fk_full=G(z)
                lg=-D(fk_full).mean()
                # Auxiliary: match mean+std of numeric dims to real batch
                if proc.num_dims:
                    idx=torch.tensor(proc.num_dims,device=DEVICE,dtype=torch.long)
                    fn=fk_full[:,idx]; rn=xb[:,idx]
                    lg += CTGAN_NUM_WEIGHT*(
                        F.mse_loss(fn.mean(0),rn.mean(0)) +
                        F.mse_loss(fn.std(0).clamp(1e-4), rn.std(0).clamp(1e-4)))
                oG.zero_grad(); lg.backward(); oG.step()
            dL+=ld.item(); gL+=lg.item()
        sG.step(); sD.step()
        if loss_history is not None: loss_history.append(gL/len(ldr))
        if ep%40==0 or ep==epochs:
            print(f"    [CTGAN]  ep {ep:4d}/{epochs}  D={dL/len(ldr):.4f}  G={gL/len(ldr):.4f}")
    G.eval()
    def sampler(n):
        parts=[]
        for s in range(0,n,2048):
            e=min(s+2048,n)
            with torch.no_grad(): out=G.sample(torch.randn(e-s,nd,device=DEVICE))
            parts.append(out.cpu().numpy())
        return np.vstack(parts)
    return sampler

# ==============================================================================
# 12.  DDPM SCHEDULER (unchanged)
# ==============================================================================

class DDPMScheduler:
    def __init__(self, T=T_STEPS, s=0.008):
        self.T=T
        steps=torch.arange(T+1,dtype=torch.float32)
        f=torch.cos(((steps/T)+s)/(1+s)*math.pi*0.5)**2; f=f/f[0]
        betas=(1.0-f[1:]/f[:-1]).clamp(1e-5,0.999)
        self.betas=betas.to(DEVICE); self.alpha_bar=f[1:].to(DEVICE)

    def q_sample(self, x0, t):
        ab=self.alpha_bar[t].view(-1,1); eps=torch.randn_like(x0)
        return ab.sqrt()*x0+(1-ab).sqrt()*eps, eps

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, cond=None):
        x=torch.randn(shape,device=device)
        ddim_start=int(self.T*0.5)
        for i in reversed(range(self.T)):
            tb=torch.full((shape[0],),i,device=device,dtype=torch.long)
            eps=model(x,tb) if cond is None else model(x,tb,cond)
            ab=self.alpha_bar[i]
            ab_prev=self.alpha_bar[i-1] if i>0 else torch.tensor(1.0,device=device)
            x0h=((x-(1-ab).sqrt()*eps)/ab.sqrt()).clamp(-5,5)
            if i < ddim_start:
                x=ab_prev.sqrt()*x0h+(1-ab_prev).sqrt()*eps
            else:
                beta=self.betas[i]
                mean=(ab_prev.sqrt()*beta/(1-ab))*x0h+(ab.sqrt()*(1-ab_prev)/(1-ab))*x
                var=(beta*(1-ab_prev)/(1-ab)).clamp(1e-20)
                x=mean+var.sqrt()*torch.randn_like(x)
        return x

# ==============================================================================
# 13.  DIFFUSION DENOISER (unchanged)
# ==============================================================================

class SinPE(nn.Module):
    def __init__(self,dim): super().__init__(); self.dim=dim
    def forward(self,t):
        h=self.dim//2
        f=torch.exp(-math.log(10000)*torch.arange(h,device=t.device)/max(h-1,1))
        a=t[:,None].float()*f[None]
        return torch.cat([a.sin(),a.cos()],dim=-1)

class FiLMBlock(nn.Module):
    def __init__(self,dim,td):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.SiLU(),nn.Dropout(0.05),nn.Linear(dim*2,dim))
        self.film=nn.Sequential(nn.Linear(td,dim*2),nn.SiLU(),nn.Linear(dim*2,dim*2))
    def forward(self,x,te):
        sc,sh=self.film(te).chunk(2,dim=-1)
        return x+self.ff(self.norm(x)*(1+sc)+sh)

class DiffNet(nn.Module):
    def __init__(self, lat_d=LATENT_DIM, hid=HIDDEN_DIM, n_blocks=6):
        super().__init__()
        self.te=nn.Sequential(SinPE(hid),nn.Linear(hid,hid*2),nn.SiLU(),nn.Linear(hid*2,hid))
        self.xp=nn.Sequential(nn.Linear(lat_d,hid),nn.LayerNorm(hid),nn.SiLU())
        self.blocks=nn.ModuleList([FiLMBlock(hid,hid) for _ in range(n_blocks)])
        self.out=nn.Sequential(nn.LayerNorm(hid),nn.Linear(hid,hid),nn.SiLU(),nn.Linear(hid,lat_d))
    def forward(self,x,t,*a):
        te=self.te(t); h=self.xp(x)
        for b in self.blocks: h=b(h,te)
        return self.out(h)

# ==============================================================================
# 14.  SHARED: encode to latent + normalise (unchanged)
# ==============================================================================

def _encode_vae(vae, X, batch=512):
    parts=[]
    with torch.no_grad():
        for s in range(0,len(X),batch):
            parts.append(vae.encode_mu(gpu(X[s:s+batch])).cpu())
    Z=torch.cat(parts)
    print(f"  [encode] {len(X)}→{tuple(Z.shape)}  μ={Z.mean():.3f} σ={Z.std():.3f}")
    return Z

def _norm_latent(Z):
    Zm=Z.mean(0,keepdim=True); Zs=Z.std(0,keepdim=True).clamp(1e-3)
    return (Z-Zm)/Zs, Zm, Zs

# ==============================================================================
# 15.  NUMERIC CALIBRATION (unchanged)
# ==============================================================================

def _calibrate_numeric(X_syn: np.ndarray, X_real: np.ndarray,
                        num_dims: list) -> np.ndarray:
    """
    Shift and scale each numeric encoded dim in X_syn so that
    mean and std match X_real.  Clips to [0,1] after.
    """
    out = X_syn.copy()
    for d in num_dims:
        r_mu = float(X_real[:,d].mean()); r_std = float(X_real[:,d].std())
        s_mu = float(out[:,d].mean());    s_std = float(out[:,d].std())
        if s_std < 1e-6: s_std = 1e-6
        out[:,d] = (out[:,d]-s_mu)/s_std * r_std + r_mu
        out[:,d] = np.clip(out[:,d], 0.0, 1.0)
    return out

# ==============================================================================
# 16.  SHARED DIFFUSION TRAINING (unchanged)
# ==============================================================================

def _train_diff_on_latent(Zn, lat_d, epochs, loss_history, tag,
                           lr_mult=2.0, n_blocks=6):
    model=DiffNet(lat_d=lat_d,hid=HIDDEN_DIM,n_blocks=n_blocks).to(DEVICE)
    ema=EMA(model); sch=DDPMScheduler(T_STEPS)
    opt=torch.optim.AdamW(model.parameters(),lr=LR*lr_mult,weight_decay=1e-4)
    T0=max(epochs//4,20)
    lrs=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T0,eta_min=LR*0.05)
    ldr=make_loader(Zn)
    for ep in range(1,epochs+1):
        model.train(); tot=0.0
        for (zb,) in ldr:
            zb=gpu(zb); t=torch.randint(0,T_STEPS,(zb.size(0),),device=DEVICE)
            zt,noise=sch.q_sample(zb,t)
            loss=F.mse_loss(model(zt,t),noise)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); lrs.step(); ema.update(model); tot+=loss.item()
        if loss_history is not None: loss_history.append(tot/len(ldr))
        if ep%50==0 or ep==epochs:
            print(f"    [{tag}]  ep {ep:4d}/{epochs}  loss={tot/len(ldr):.5f}")
    ema.apply(model); model.eval()
    return model, sch


# ==============================================================================
# 17.  BASELINE – DIFFUSION  (CVAE latent + flat DiffNet)
# ==============================================================================

def train_diffusion(X, proc, epochs=EPOCHS_DIFF, loss_history=None):
    """Plain latent-diffusion baseline: lightweight CVAE pre-train + flat DiffNet."""
    vae = CVAE(X.shape[1], proc.cat_groups, proc.num_dims).to(DEVICE)
    _train_vae_core(vae, X, proc, EPOCHS_VAE_PRETRAIN, cyc_kl=True, tag="Diff/VAE")
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    Z = _encode_vae(vae, X); Zn, Zm, Zs = _norm_latent(Z)
    vae.cpu()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    model, sch = _train_diff_on_latent(Zn, Zn.shape[1], epochs, loss_history, "Diffusion")
    def sampler(n):
        vae.to(DEVICE); Zm_d = Zm.to(DEVICE); Zs_d = Zs.to(DEVICE); parts = []
        for s in range(0, n, 256):
            e = min(s + 256, n)
            z = sch.p_sample_loop(model, (e - s, Zn.shape[1]), DEVICE) * Zs_d + Zm_d
            parts.append(vae.dec.sample(z, CAT_TEMP).cpu().numpy())
        vae.cpu()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        return np.vstack(parts).astype(np.float32)
    return sampler

# ==============================================================================
# 19.  HiDe‑Tab (proposed model)
# ==============================================================================

# ---- 19.1 Disentangled CVAE with TC loss ----
class DisentangledEncoder(nn.Module):
    """Three independent encoders for z_s, z_v, z_c."""
    def __init__(self, in_d, hid, lat_s, lat_v, lat_c):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(in_d, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU()
        )
        # Heads for each latent
        self.mu_s = nn.Linear(hid, lat_s); self.lv_s = nn.Linear(hid, lat_s)
        self.mu_v = nn.Linear(hid, lat_v); self.lv_v = nn.Linear(hid, lat_v)
        self.mu_c = nn.Linear(hid, lat_c); self.lv_c = nn.Linear(hid, lat_c)

    def forward(self, x):
        h = self.shared(x)
        mu_s, lv_s = self.mu_s(h), self.lv_s(h).clamp(-4,4)
        mu_v, lv_v = self.mu_v(h), self.lv_v(h).clamp(-4,4)
        mu_c, lv_c = self.mu_c(h), self.lv_c(h).clamp(-4,4)
        return (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5*lv) * torch.randn_like(mu)

class DisentangledDecoder(nn.Module):
    """Decodes concatenated [z_s, z_v, z_c] to data space."""
    def __init__(self, lat_total, hid, cat_groups, num_dims, total_dims):
        super().__init__()
        self.decoder = MultiHeadDecoder(lat_total, hid, cat_groups, num_dims, total_dims)

    def forward(self, z):
        return self.decoder(z)

    def sample(self, z, temp=CAT_TEMP):
        return self.decoder.sample(z, temp)

class DisentangledCVAE(nn.Module):
    def __init__(self, in_d, cat_groups, num_dims, hid=HIDDEN_DIM,
                 lat_s=16, lat_v=32, lat_c=16):
        super().__init__()
        self.enc = DisentangledEncoder(in_d, hid, lat_s, lat_v, lat_c)
        self.dec = DisentangledDecoder(lat_s+lat_v+lat_c, hid, cat_groups, num_dims, in_d)
        self.lat_s_dim = lat_s; self.lat_v_dim = lat_v; self.lat_c_dim = lat_c
        self.total_lat = lat_s+lat_v+lat_c

    def forward(self, x):
        (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c) = self.enc(x)
        z_s = self.enc.reparameterize(mu_s, lv_s)
        z_v = self.enc.reparameterize(mu_v, lv_v)
        z_c = self.enc.reparameterize(mu_c, lv_c)
        z = torch.cat([z_s, z_v, z_c], dim=1)
        recon = self.dec(z)
        return recon, (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c), (z_s, z_v, z_c)

    def encode(self, x):
        """Return concatenated latent (mu) and separate latents."""
        (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c) = self.enc(x)
        return torch.cat([mu_s, mu_v, mu_c], dim=1), (mu_s, mu_v, mu_c), (lv_s, lv_v, lv_c)

    @torch.no_grad()
    def sample_latents(self, n):
        z_s = torch.randn(n, self.lat_s_dim, device=DEVICE)
        z_v = torch.randn(n, self.lat_v_dim, device=DEVICE)
        z_c = torch.randn(n, self.lat_c_dim, device=DEVICE)
        return torch.cat([z_s, z_v, z_c], dim=1), (z_s, z_v, z_c)

def tc_loss_from_groups(z_groups):
    """
    Approximate total correlation loss as sum of pairwise covariances.
    z_groups: list of tensors (N, d_i)
    """
    loss = 0.0
    for i in range(len(z_groups)):
        for j in range(i+1, len(z_groups)):
            # Center
            zi = z_groups[i] - z_groups[i].mean(0, keepdim=True)
            zj = z_groups[j] - z_groups[j].mean(0, keepdim=True)
            # Cross-covariance matrix
            cross_cov = (zi.T @ zj) / (zi.size(0)-1)
            loss += cross_cov.pow(2).sum()
    return loss / max(len(z_groups), 1)

def train_disentangled_vae(X, proc, epochs=EPOCHS_VAE, lat_s=16, lat_v=32, lat_c=16,
                           beta_kl=0.1, lambda_tc=0.05, loss_history=None):
    model = DisentangledCVAE(X.shape[1], proc.cat_groups, proc.num_dims,
                             hid=HIDDEN_DIM, lat_s=lat_s, lat_v=lat_v, lat_c=lat_c).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    ldr = make_loader(X)
    best_loss = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        total_recon = 0.0
        total_tc = 0.0
        total_kl = 0.0
        for (xb,) in ldr:
            xb = gpu(xb)
            recon, (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c), (z_s, z_v, z_c) = model(xb)

            # Reconstruction loss
            recon_loss = composite_loss(recon, xb, proc.cat_groups, proc.num_dims)

            # KL divergences for each group
            kl_s = -0.5 * torch.mean(1 + lv_s - mu_s.pow(2) - lv_s.exp())
            kl_v = -0.5 * torch.mean(1 + lv_v - mu_v.pow(2) - lv_v.exp())
            kl_c = -0.5 * torch.mean(1 + lv_c - mu_c.pow(2) - lv_c.exp())
            kl_total = kl_s + kl_v + kl_c

            # Total Correlation penalty (decorrelation)
            tc_penalty = tc_loss_from_groups([z_s, z_v, z_c])

            loss = recon_loss + beta_kl * kl_total + lambda_tc * tc_penalty
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_recon += recon_loss.item()
            total_tc += tc_penalty.item()
            total_kl += kl_total.item()

        sch.step()
        avg_loss = (total_recon + beta_kl*total_kl + lambda_tc*total_tc) / len(ldr)
        if loss_history is not None:
            loss_history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0 or ep == epochs:
            print(f"    [DisentangledVAE] ep {ep:4d}/{epochs}  recon={total_recon/len(ldr):.4f}  "
                  f"KL={total_kl/len(ldr):.3f}  TC={total_tc/len(ldr):.4f}")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model

# ---- 19.2 Two‑stage diffusion models ----
class DiffNetTransformer(nn.Module):
    """Transformer‑based denoiser for (z_s, z_c)."""
    def __init__(self, lat_d, hid=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.time_mlp = nn.Sequential(SinPE(hid), nn.Linear(hid, hid*2), nn.SiLU(), nn.Linear(hid*2, hid))
        self.input_proj = nn.Linear(lat_d, hid)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid, nhead=n_heads,
                                                    dim_feedforward=hid*2, dropout=0.1,
                                                    batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hid, lat_d)

    def forward(self, z, t):
        t_emb = self.time_mlp(t).unsqueeze(1)  # (B,1,hid)
        h = self.input_proj(z).unsqueeze(1)    # (B,1,hid)
        h = h + t_emb
        h = self.transformer(h)
        return self.out_proj(h.squeeze(1))

class DiffNetMLPCond(nn.Module):
    """MLP denoiser for z_v conditioned on (z_s, z_c)."""
    def __init__(self, lat_v, cond_dim, hid=HIDDEN_DIM, n_blocks=4):
        super().__init__()
        self.time_mlp = nn.Sequential(SinPE(hid), nn.Linear(hid, hid*2), nn.SiLU(), nn.Linear(hid*2, hid))
        self.cond_proj = nn.Linear(cond_dim, hid)
        self.input_proj = nn.Linear(lat_v, hid)
        self.blocks = nn.ModuleList([FiLMBlock(hid, hid) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, lat_v))

    def forward(self, z_v, t, z_sc):
        t_emb = self.time_mlp(t)
        cond = self.cond_proj(z_sc)
        h = self.input_proj(z_v) + cond
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out(h)


class DiffNetCNN(nn.Module):
    """
    1D CNN denoiser for z_v conditioned on (z_s, z_c).
    Input: z_v (B, lat_v), t (B), z_sc (B, cond_dim)
    Output: predicted noise (B, lat_v)
    """

    def __init__(self, lat_v, cond_dim, hid=256, n_blocks=4):
        super().__init__()
        self.lat_v = lat_v
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinPE(hid), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, hid)
        )
        # Conditioning projection for FiLM (scale & shift)
        self.cond_film = nn.Sequential(
            nn.Linear(cond_dim, hid), nn.SiLU(), nn.Linear(hid, hid * 2)
        )
        # Input projection: (B, 1, lat_v) -> (B, hid, lat_v)
        self.input_conv = nn.Conv1d(1, hid, kernel_size=3, padding=1)

        # Convolutional blocks with FiLM
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hid, hid, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hid),
                    nn.SiLU(),
                )
            )
        # Final convolution to reduce channels to 1
        self.out_conv = nn.Sequential(
            nn.Conv1d(hid, hid // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hid // 2, 1, kernel_size=3, padding=1)
        )
        # Final linear projection to lat_v (if needed)
        self.final_proj = nn.Linear(lat_v, lat_v)

    def forward(self, z_v, t, z_sc):
        # z_v: (B, lat_v) -> (B, 1, lat_v)
        x = z_v.unsqueeze(1)  # (B, 1, lat_v)

        # Time embedding for FiLM (will be used per block)
        t_emb = self.time_mlp(t)  # (B, hid)

        # Conditioning embedding for FiLM (scale and shift)
        film_params = self.cond_film(z_sc)  # (B, hid*2)
        film_scale, film_shift = film_params.chunk(2, dim=-1)  # each (B, hid)

        # Input convolution
        h = self.input_conv(x)  # (B, hid, lat_v)

        # Apply each block with FiLM conditioning
        for blk in self.blocks:
            h_res = h
            h = blk(h)
            # FiLM: scale and shift along channel dimension
            # Expand film_scale/shift to (B, hid, 1) for broadcasting
            scale = film_scale.unsqueeze(-1)
            shift = film_shift.unsqueeze(-1)
            h = h * (1 + scale) + shift
            h = h + h_res  # residual connection

        # Final convolutions and projection
        out = self.out_conv(h).squeeze(1)  # (B, lat_v)
        out = self.final_proj(out)
        return out

# ---- 19.3 HiDe‑Tab main training function ----
def train_hidetab(X, proc, epochs_diff=EPOCHS_DIFF, epochs_ft=EPOCHS_FINETUNE,
                  lat_s=24, lat_v=64, lat_c=24, loss_history=None):
    """
    HiDe‑Tab: Hierarchical Disentangled Diffusion for Tabular Data.
    Stage B now uses a 1D CNN instead of MLP.
    """
    print("\n  [HiDe-Tab] Stage 1 – Training Disentangled VAE")
    vae = train_disentangled_vae(X, proc, epochs=EPOCHS_VAE_PRETRAIN,
                                 lat_s=lat_s, lat_v=lat_v, lat_c=lat_c)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Encode data to obtain latents
    print("  [HiDe-Tab] Encoding training data into latents")
    Z_sc_list, Z_v_list = [], []
    with torch.no_grad():
        for batch in make_loader(X, batch=256):
            xb = gpu(batch[0])
            _, (mu_s, mu_v, mu_c), _ = vae.encode(xb)
            z_sc = torch.cat([mu_s, mu_c], dim=1)
            Z_sc_list.append(z_sc.cpu())
            Z_v_list.append(mu_v.cpu())
    Z_sc = torch.cat(Z_sc_list, dim=0)
    Z_v = torch.cat(Z_v_list, dim=0)
    # Normalise latents
    Z_sc_mean, Z_sc_std = Z_sc.mean(0, keepdim=True), Z_sc.std(0, keepdim=True).clamp(1e-3)
    Z_v_mean, Z_v_std = Z_v.mean(0, keepdim=True), Z_v.std(0, keepdim=True).clamp(1e-3)
    Zn_sc = (Z_sc - Z_sc_mean) / Z_sc_std
    Zn_v = (Z_v - Z_v_mean) / Z_v_std
    print(f"    Latent dimensions: z_sc={Zn_sc.shape[1]}, z_v={Zn_v.shape[1]}")

    # Stage A diffusion on (z_s, z_c)
    print("  [HiDe-Tab] Stage A – Training macro diffusion (Transformer)")
    model_A = DiffNetTransformer(lat_d=Zn_sc.shape[1]).to(DEVICE)
    ema_A = EMA(model_A)
    sch_A = DDPMScheduler(T_STEPS)  # T=500
    opt_A = torch.optim.AdamW(model_A.parameters(), lr=LR*2.0, weight_decay=1e-4)
    lr_sch_A = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_A, T_0=50, eta_min=LR*0.05)

    for ep in range(1, epochs_diff+1):
        model_A.train()
        total = 0.0
        for (z_sc_batch,) in make_loader(Zn_sc):
            z_sc_batch = gpu(z_sc_batch)
            t = torch.randint(0, T_STEPS, (z_sc_batch.size(0),), device=DEVICE)
            zt, noise = sch_A.q_sample(z_sc_batch, t)
            loss = F.mse_loss(model_A(zt, t), noise)
            opt_A.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            opt_A.step()
            lr_sch_A.step()
            ema_A.update(model_A)
            total += loss.item()
        if loss_history is not None:
            loss_history.append(total / len(Zn_sc))
        if ep % 50 == 0 or ep == epochs_diff:
            print(f"      Stage A ep {ep:4d}/{epochs_diff}  loss={total/len(Zn_sc):.5f}")
    ema_A.apply(model_A)
    model_A.eval()

    # Stage B diffusion on z_v conditioned on (z_s, z_c) – NOW USING CNN
    print("  [HiDe-Tab] Stage B – Training micro diffusion (1D CNN with conditioning)")
    cond_dim = Zn_sc.shape[1]
    model_B = DiffNetCNN(lat_v=Zn_v.shape[1], cond_dim=cond_dim, hid=256, n_blocks=4).to(DEVICE)
    ema_B = EMA(model_B)
    sch_B = DDPMScheduler(T_STEPS//2)  # T=200
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=LR, weight_decay=1e-4)
    lr_sch_B = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_B, T_0=30, eta_min=LR*0.05)

    # Create loader with both (z_v, z_sc)
    dataset = torch.utils.data.TensorDataset(Zn_v, Zn_sc)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type=="cuda"))
    for ep in range(1, epochs_diff+1):
        model_B.train()
        total = 0.0
        for z_v_batch, z_sc_batch in loader:
            z_v_batch = gpu(z_v_batch)
            z_sc_batch = gpu(z_sc_batch)
            t = torch.randint(0, T_STEPS//2, (z_v_batch.size(0),), device=DEVICE)
            zt, noise = sch_B.q_sample(z_v_batch, t)
            loss = F.mse_loss(model_B(zt, t, z_sc_batch), noise)
            opt_B.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
            opt_B.step()
            lr_sch_B.step()
            ema_B.update(model_B)
            total += loss.item()
        if loss_history is not None:
            loss_history.append(total / len(loader))
        if ep % 50 == 0 or ep == epochs_diff:
            print(f"      Stage B ep {ep:4d}/{epochs_diff}  loss={total/len(loader):.5f}")
    ema_B.apply(model_B)
    model_B.eval()

    # ── Sampler (matches Algorithm 1: no PGD) ────────────────────────────────
    def hidetab_sampler(n):
        vae.to(DEVICE); model_A.to(DEVICE); model_B.to(DEVICE)
        Z_sc_mean_d = Z_sc_mean.to(DEVICE); Z_sc_std_d = Z_sc_std.to(DEVICE)
        Z_v_mean_d  = Z_v_mean.to(DEVICE);  Z_v_std_d  = Z_v_std.to(DEVICE)
        all_samples = []

        for start in range(0, n, 256):
            cur_batch = min(256, n - start)

            # Stage A: sample z_sc_norm ~ p(z_sc) via Transformer denoiser (T=300)
            z_sc_norm = sch_A.p_sample_loop(model_A, (cur_batch, Zn_sc.shape[1]), DEVICE)
            # Denormalize for decoding; keep normalized copy for Stage B conditioning
            # (model_B was trained on normalized z_sc, so pass z_sc_norm as cond)
            z_sc = z_sc_norm * Z_sc_std_d + Z_sc_mean_d

            # Stage B: sample z_v_norm ~ p(z_v | z_sc) via 1D-CNN denoiser (T=T/2)
            # cond = normalized z_sc (matches training distribution of model_B)
            z_v_norm = sch_B.p_sample_loop(model_B, (cur_batch, Zn_v.shape[1]),
                                            DEVICE, cond=z_sc_norm)
            z_v = z_v_norm * Z_v_std_d + Z_v_mean_d

            # Decode: z_full must follow VAE decoder order [z_s; z_v; z_c]
            # z_sc = [z_s; z_c] → split back
            z_s = z_sc[:, :Zn_sc.shape[1]//2]      # first lat_s dims
            z_c = z_sc[:, Zn_sc.shape[1]//2:]      # last lat_c dims
            z_full = torch.cat([z_s, z_v, z_c], dim=1)
            x_syn = vae.dec.sample(z_full, CAT_TEMP)
            all_samples.append(x_syn.cpu().numpy())

        vae.cpu(); model_A.cpu(); model_B.cpu()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return np.vstack(all_samples).astype(np.float32)

    return hidetab_sampler

# ==============================================================================
# 19.5  HiDe-Tab ABLATION VARIANTS
#   Toggles: TC penalty │ hierarchy depth (1/2/3-stage) │ FiLM conditioning │ PGD
# ==============================================================================

HIDETAB_ABLATION_CONFIGS = {
    # name               lambda_tc  stage_mode        use_film
    "HiDeTab_noTC":   dict(lambda_tc=0.0,  stage_mode="two_stage",   use_film=True),
    "HiDeTab_1Stage": dict(lambda_tc=0.05, stage_mode="one_stage",   use_film=True),
    "HiDeTab_3Stage": dict(lambda_tc=0.05, stage_mode="three_stage", use_film=True),
    "HiDeTab_noFiLM": dict(lambda_tc=0.05, stage_mode="two_stage",   use_film=False),
}
# "HiDe_Tab" (the full model) is registered separately in _dispatch.
# HiDeTab_Full is an alias for HiDe_Tab; run HiDe_Tab instead.


def _train_cond_diffusion_on_latent(Zn_target, Zn_cond, target_dim, cond_dim, epochs,
                                     loss_history, tag, T=T_STEPS, hid=256, n_blocks=4,
                                     zero_cond=False):
    """Generic conditional 1D-CNN diffusion trainer (used for any z_target | z_cond pair)."""
    model = DiffNetCNN(lat_v=target_dim, cond_dim=cond_dim, hid=hid, n_blocks=n_blocks).to(DEVICE)
    ema = EMA(model)
    sch = DDPMScheduler(T)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    T0 = max(epochs // 4, 20)
    lrs = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T0, eta_min=LR * 0.05)
    dataset = torch.utils.data.TensorDataset(Zn_target, Zn_cond)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         pin_memory=(DEVICE.type == "cuda"))
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for zb, cb in loader:
            zb = gpu(zb); cb = gpu(cb)
            if zero_cond:
                cb = torch.zeros_like(cb)          # <-- ablates the conditioning signal
            t = torch.randint(0, T, (zb.size(0),), device=DEVICE)
            zt, noise = sch.q_sample(zb, t)
            loss = F.mse_loss(model(zt, t, cb), noise)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); lrs.step(); ema.update(model); tot += loss.item()
        if loss_history is not None:
            loss_history.append(tot / len(loader))
        if ep % 50 == 0 or ep == epochs:
            print(f"      [{tag}]  ep {ep:4d}/{epochs}  loss={tot/len(loader):.5f}")
    ema.apply(model); model.eval()
    return model, sch


def _reverse_loop(model, sch, cond, shape):
    """Plain DDPM reverse loop with optional FiLM conditioning. No PGD."""
    x = torch.randn(shape, device=DEVICE)
    for i in reversed(range(sch.T)):
        t   = torch.full((shape[0],), i, device=DEVICE, dtype=torch.long)
        eps = model(x, t) if cond is None else model(x, t, cond)
        ab      = sch.alpha_bar[i]
        ab_prev = sch.alpha_bar[i - 1] if i > 0 else torch.tensor(1.0, device=DEVICE)
        x0h = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-5, 5)
        if i > 0:
            beta = sch.betas[i]
            mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0h + \
                   (ab.sqrt() * (1 - ab_prev) / (1 - ab)) * x
            var  = (beta * (1 - ab_prev) / (1 - ab)).clamp(1e-20)
            x    = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0h
    return x


def train_hidetab_variant(X, proc, name="HiDeTab_Variant",
                           lambda_tc=0.05, stage_mode="two_stage",
                           use_film=True,
                           epochs_diff=EPOCHS_DIFF,
                           lat_s=24, lat_v=64, lat_c=24,
                           loss_history=None):
    """
    Unified HiDe-Tab ablation trainer. stage_mode in {"one_stage","two_stage","three_stage"}.
    Latent concatenation order is ALWAYS [z_s, z_v, z_c] to match DisentangledCVAE's decoder.
    """
    assert stage_mode in ("one_stage", "two_stage", "three_stage")
    print(f"\n  [{name}]  lambda_tc={lambda_tc}  stage_mode={stage_mode}  "
          f"FiLM={use_film}")

    # ---- Stage 1: disentangled VAE (TC penalty toggled here) ----
    vae = train_disentangled_vae(X, proc, epochs=EPOCHS_VAE_PRETRAIN,
                                  lat_s=lat_s, lat_v=lat_v, lat_c=lat_c,
                                  lambda_tc=lambda_tc, loss_history=None)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)

    # ---- Encode dataset into the three latent partitions ----
    Zs_l, Zv_l, Zc_l = [], [], []
    with torch.no_grad():
        for batch in make_loader(X, batch=256):
            xb = gpu(batch[0])
            _, (mu_s, mu_v, mu_c), _ = vae.encode(xb)
            Zs_l.append(mu_s.cpu()); Zv_l.append(mu_v.cpu()); Zc_l.append(mu_c.cpu())
    Zs, Zv, Zc = torch.cat(Zs_l), torch.cat(Zv_l), torch.cat(Zc_l)

    def _norm(Z):
        m = Z.mean(0, keepdim=True); s = Z.std(0, keepdim=True).clamp(1e-3)
        return (Z - m) / s, m, s
    Zns, Zsm, Zss = _norm(Zs)
    Znv, Zvm, Zvs = _norm(Zv)
    Znc, Zcm, Zcs = _norm(Zc)

    Zsm_d, Zss_d = Zsm.to(DEVICE), Zss.to(DEVICE)
    Zvm_d, Zvs_d = Zvm.to(DEVICE), Zvs.to(DEVICE)
    Zcm_d, Zcs_d = Zcm.to(DEVICE), Zcs.to(DEVICE)

    # ================= ONE-STAGE: flat diffusion over [z_s,z_v,z_c] =================
    if stage_mode == "one_stage":
        Zn_full = torch.cat([Zns, Znv, Znc], dim=1)
        model, sch = _train_diff_on_latent(Zn_full, Zn_full.shape[1], epochs_diff,
                                            loss_history, f"{name}/flat")

        def sampler(n):
            vae.to(DEVICE); parts = []
            for s0 in range(0, n, 256):
                bs = min(256, n - s0)
                z_norm = _reverse_loop(model, sch, cond=None, shape=(bs, Zn_full.shape[1]))
                # Denormalize each partition (order in Zn_full is [z_s, z_v, z_c])
                z_s = z_norm[:, :lat_s]               * Zss_d + Zsm_d
                z_v = z_norm[:, lat_s:lat_s+lat_v]    * Zvs_d + Zvm_d
                z_c = z_norm[:, lat_s+lat_v:]          * Zcs_d + Zcm_d
                z_full = torch.cat([z_s, z_v, z_c], dim=1)
                parts.append(vae.dec.sample(z_full, CAT_TEMP).cpu().numpy())
            vae.cpu()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            return np.vstack(parts).astype(np.float32)
        return sampler

    # ================= TWO-STAGE: z_sc macro (Transformer) → z_v micro | z_sc (CNN) =================
    if stage_mode == "two_stage":
        Zn_sc = torch.cat([Zns, Znc], dim=1)
        Zsc_m = torch.cat([Zsm, Zcm], dim=1).to(DEVICE)
        Zsc_s = torch.cat([Zss, Zcs], dim=1).to(DEVICE)

        model_A = DiffNetTransformer(lat_d=Zn_sc.shape[1]).to(DEVICE)
        ema_A = EMA(model_A); sch_A = DDPMScheduler(T_STEPS)
        opt_A = torch.optim.AdamW(model_A.parameters(), lr=LR*2.0, weight_decay=1e-4)
        lr_A = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_A, max(epochs_diff//6,10), eta_min=LR*0.05)
        for ep in range(1, epochs_diff+1):
            model_A.train(); tot = 0.0
            for (zb,) in make_loader(Zn_sc):
                zb = gpu(zb); t = torch.randint(0, T_STEPS, (zb.size(0),), device=DEVICE)
                zt, noise = sch_A.q_sample(zb, t)
                loss = F.mse_loss(model_A(zt, t), noise)
                opt_A.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
                opt_A.step(); lr_A.step(); ema_A.update(model_A); tot += loss.item()
            if loss_history is not None: loss_history.append(tot/len(Zn_sc))
            if ep % 50 == 0 or ep == epochs_diff:
                print(f"      [{name}/StageA] ep {ep:4d}/{epochs_diff}  loss={tot/len(Zn_sc):.5f}")
        ema_A.apply(model_A); model_A.eval()

        model_B, sch_B = _train_cond_diffusion_on_latent(
            Znv, Zn_sc, target_dim=lat_v, cond_dim=Zn_sc.shape[1],
            epochs=epochs_diff, loss_history=loss_history, tag=f"{name}/StageB",
            T=T_STEPS//2, zero_cond=not use_film)

        def sampler(n):
            vae.to(DEVICE); parts = []
            for s0 in range(0, n, 256):
                bs = min(256, n - s0)
                # Stage A: sample normalized z_sc via Transformer
                z_sc_norm = sch_A.p_sample_loop(model_A, (bs, Zn_sc.shape[1]), DEVICE)
                z_sc = z_sc_norm * Zsc_s + Zsc_m
                # Stage B: sample z_v conditioned on normalized z_sc
                # (model_B was trained on normalized z_sc as conditioning signal)
                cond_B = z_sc_norm if use_film else torch.zeros_like(z_sc_norm)
                zv_norm = _reverse_loop(model_B, sch_B, cond=cond_B, shape=(bs, lat_v))
                zv = zv_norm * Zvs_d + Zvm_d
                # Decoder expects [z_s; z_v; z_c]; z_sc = [z_s; z_c]
                z_s = z_sc[:, :lat_s]
                z_c = z_sc[:, lat_s:]
                z_full = torch.cat([z_s, zv, z_c], dim=1)
                parts.append(vae.dec.sample(z_full, CAT_TEMP).cpu().numpy())
            vae.cpu()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            return np.vstack(parts).astype(np.float32)
        return sampler

    # ================= THREE-STAGE: z_s → z_c|z_s → z_v|z_s,z_c =================
    if stage_mode == "three_stage":
        model_A = DiffNetTransformer(lat_d=lat_s).to(DEVICE)
        ema_A = EMA(model_A); sch_A = DDPMScheduler(T_STEPS)
        opt_A = torch.optim.AdamW(model_A.parameters(), lr=LR*2.0, weight_decay=1e-4)
        lr_A = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_A, max(epochs_diff//6,10), eta_min=LR*0.05)
        for ep in range(1, epochs_diff+1):
            model_A.train(); tot = 0.0
            for (zb,) in make_loader(Zns):
                zb = gpu(zb); t = torch.randint(0, T_STEPS, (zb.size(0),), device=DEVICE)
                zt, noise = sch_A.q_sample(zb, t)
                loss = F.mse_loss(model_A(zt, t), noise)
                opt_A.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
                opt_A.step(); lr_A.step(); ema_A.update(model_A); tot += loss.item()
            if loss_history is not None: loss_history.append(tot/len(Zns))
            if ep % 50 == 0 or ep == epochs_diff:
                print(f"      [{name}/StageA z_s] ep {ep:4d}/{epochs_diff}  loss={tot/len(Zns):.5f}")
        ema_A.apply(model_A); model_A.eval()

        model_B, sch_B = _train_cond_diffusion_on_latent(
            Znc, Zns, target_dim=lat_c, cond_dim=lat_s,
            epochs=epochs_diff, loss_history=loss_history, tag=f"{name}/StageB z_c|z_s",
            T=int(T_STEPS*0.66), zero_cond=not use_film)

        Zn_sc_cat = torch.cat([Zns, Znc], dim=1)
        model_C, sch_C = _train_cond_diffusion_on_latent(
            Znv, Zn_sc_cat, target_dim=lat_v, cond_dim=lat_s+lat_c,
            epochs=epochs_diff, loss_history=loss_history, tag=f"{name}/StageC z_v|z_s,z_c",
            T=T_STEPS//2, zero_cond=not use_film)

        def sampler(n):
            vae.to(DEVICE); parts = []
            for s0 in range(0, n, 256):
                bs = min(256, n - s0)
                # Stage A: sample z_s unconditionally
                zs_norm = sch_A.p_sample_loop(model_A, (bs, lat_s), DEVICE)
                zs = zs_norm * Zss_d + Zsm_d
                # Stage B: sample z_c | z_s (cond on normalized z_s)
                cond_B = zs_norm if use_film else torch.zeros_like(zs_norm)
                zc_norm = _reverse_loop(model_B, sch_B, cond=cond_B, shape=(bs, lat_c))
                zc = zc_norm * Zcs_d + Zcm_d
                # Stage C: sample z_v | z_s, z_c (cond on normalized [z_s; z_c])
                cond_C = (torch.cat([zs_norm, zc_norm], dim=1) if use_film
                          else torch.zeros(bs, lat_s + lat_c, device=DEVICE))
                zv_norm = _reverse_loop(model_C, sch_C, cond=cond_C, shape=(bs, lat_v))
                zv = zv_norm * Zvs_d + Zvm_d
                # Decoder expects [z_s; z_v; z_c]
                z_full = torch.cat([zs, zv, zc], dim=1)
                parts.append(vae.dec.sample(z_full, CAT_TEMP).cpu().numpy())
            vae.cpu()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            return np.vstack(parts).astype(np.float32)
        return sampler


# ==============================================================================
# 20.  TABDIFF (ICLR 2025) – mixed-type diffusion model
# ==============================================================================
# References:
#   - Paper: https://arxiv.org/abs/2410.20626
#   - GitHub: https://github.com/MinkaiXu/TabDiff
#   - ICLR 2025

class TabDiffEncoder(nn.Module):
    """Simple encoder for TabDiff – maps mixed-type data to a latent representation."""
    def __init__(self, in_d, hid, lat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid, lat)
        )
    def forward(self, x):
        return self.net(x)

class TabDiffDecoder(nn.Module):
    """Decoder for TabDiff."""
    def __init__(self, lat, hid, cat_groups, num_dims, total_dims):
        super().__init__()
        self.cat_groups = cat_groups
        self.num_dims = num_dims
        self.total = total_dims
        self.trunk = nn.Sequential(
            nn.Linear(lat, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU()
        )
        self.cat_heads = nn.ModuleList([nn.Linear(hid, K) for _, _, K in cat_groups])
        if num_dims:
            self.num_head = nn.Linear(hid, len(num_dims))
        else:
            self.num_head = None

    def forward(self, z):
        h = self.trunk(z)
        out = torch.zeros(z.size(0), self.total, device=z.device)
        for head, (s, e, K) in zip(self.cat_heads, self.cat_groups):
            out[:, s:e] = head(h)
        if self.num_head is not None:
            idx = torch.tensor(self.num_dims, device=z.device, dtype=torch.long)
            out[:, idx] = torch.sigmoid(self.num_head(h))
        return out

    @torch.no_grad()
    def sample(self, z, temperature=CAT_TEMP):
        h = self.trunk(z)
        out = torch.zeros(z.size(0), self.total, device=z.device)
        for head, (s, e, K) in zip(self.cat_heads, self.cat_groups):
            probs = F.softmax(head(h) / max(temperature, 1e-6), dim=-1)
            oh = F.one_hot(torch.multinomial(probs, 1).squeeze(-1), K).float()
            out[:, s:e] = oh
        if self.num_head is not None:
            idx = torch.tensor(self.num_dims, device=z.device, dtype=torch.long)
            out[:, idx] = torch.sigmoid(self.num_head(h))
        return out

class TabDiff(nn.Module):
    """
    TabDiff: Mixed-type Diffusion Model for Tabular Data Generation.
    Based on the architecture described in the ICLR 2025 paper.
    """
    def __init__(self, in_d, cat_groups, num_dims, hid=HIDDEN_DIM, lat=LATENT_DIM):
        super().__init__()
        self.enc = TabDiffEncoder(in_d, hid, lat)
        self.dec = TabDiffDecoder(lat, hid, cat_groups, num_dims, in_d)
        self.lat = lat

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z

    @torch.no_grad()
    def sample(self, n, latent):
        z = latent if latent is not None else torch.randn(n, self.lat, device=DEVICE)
        return self.dec.sample(z, CAT_TEMP).cpu().numpy()

def train_tabdiff(X, proc, epochs=EPOCHS_DIFF, loss_history=None):
    """
    Train TabDiff model.
    """
    print("  [TabDiff] Training mixed-type diffusion model")
    model = TabDiff(X.shape[1], proc.cat_groups, proc.num_dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    ldr = make_loader(X)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for (xb,) in ldr:
            xb = gpu(xb)
            recon, z = model(xb)
            loss = composite_loss(recon, xb, proc.cat_groups, proc.num_dims)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sch.step()
        avg_loss = total_loss / len(ldr)
        if loss_history is not None:
            loss_history.append(avg_loss)
        if ep % 50 == 0 or ep == epochs:
            print(f"      ep {ep:4d}/{epochs}  loss={avg_loss:.5f}")

    model.eval()
    def sampler(n):
        return model.sample(n, latent=None)
    return sampler


# ==============================================================================
# 21.  TABSYN (ICLR 2024 Oral) – score-based diffusion in latent space
# ==============================================================================
# References:
#   - Paper: Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space (ICLR 2024 Oral)
#   - GitHub: https://github.com/amazon-science/tabsyn

class TabSynVAE(nn.Module):
    """
    TabSyn VAE – encodes mixed-type data into a continuous latent space.
    """
    def __init__(self, in_d, cat_groups, num_dims, hid=HIDDEN_DIM, lat=LATENT_DIM):
        super().__init__()
        self.enc = CVAEEnc(in_d, hid, lat)   # reuse encoder
        self.dec = MultiHeadDecoder(lat, hid, cat_groups, num_dims, in_d)
        self.lat = lat

    def _reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def forward(self, x):
        mu, lv = self.enc(x)
        z = self._reparameterize(mu, lv)
        recon = self.dec(z)
        return recon, mu, lv, z

    @torch.no_grad()
    def encode(self, x):
        mu, _ = self.enc(x)
        return mu

    @torch.no_grad()
    def decode(self, z):
        return self.dec.sample(z, CAT_TEMP)

def train_tabsyn_vae(X, proc, epochs=EPOCHS_VAE_PRETRAIN, lat=LATENT_DIM, loss_history=None):
    """
    Pre-train VAE for TabSyn.
    """
    model = TabSynVAE(X.shape[1], proc.cat_groups, proc.num_dims, hid=HIDDEN_DIM, lat=lat).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    ldr = make_loader(X)
    best_loss = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for (xb,) in ldr:
            xb = gpu(xb)
            recon, mu, lv, _ = model(xb)
            recon_loss = composite_loss(recon, xb, proc.cat_groups, proc.num_dims)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = recon_loss + 0.1 * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sch.step()
        avg_loss = total_loss / len(ldr)
        if loss_history is not None:
            loss_history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0 or ep == epochs:
            print(f"    [TabSyn/VAE] ep {ep:4d}/{epochs}  loss={avg_loss:.5f}")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model

def train_tabsyn(X, proc, epochs=EPOCHS_DIFF, loss_history=None):
    """
    Train TabSyn: VAE + diffusion on latent space.
    """
    print("  [TabSyn] Stage 1 – Pre-training VAE")
    vae = train_tabsyn_vae(X, proc, epochs=EPOCHS_VAE_PRETRAIN, loss_history=loss_history)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Encode data to latents
    print("  [TabSyn] Encoding data to latent space")
    Z_list = []
    with torch.no_grad():
        for batch in make_loader(X, batch=256):
            xb = gpu(batch[0])
            z = vae.encode(xb)
            Z_list.append(z.cpu())
    Z = torch.cat(Z_list, dim=0)
    Zn, Zm, Zs = _norm_latent(Z)
    print(f"    Latent dimension: {Zn.shape[1]}")

    # Train diffusion model on latent
    print("  [TabSyn] Stage 2 – Training diffusion model on latent space")
    diff_model, sch = _train_diff_on_latent(Zn, Zn.shape[1], epochs, loss_history, "TabSyn")

    def sampler(n):
        vae.to(DEVICE)
        Zm_d = Zm.to(DEVICE)
        Zs_d = Zs.to(DEVICE)
        parts = []
        for start in range(0, n, 256):
            end = min(start+256, n)
            z = sch.p_sample_loop(diff_model, (end-start, Zn.shape[1]), DEVICE) * Zs_d + Zm_d
            x_syn = vae.decode(z).cpu().numpy()
            parts.append(x_syn)
        vae.cpu()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return np.vstack(parts).astype(np.float32)
    return sampler


# ==============================================================================
# 22.  EVALUATION (unchanged)
# ==============================================================================

def _strip(df):
    out=df.copy()
    for c in out.columns:
        if hasattr(out[c],"cat"): out[c]=out[c].astype(object)
    return out

def _jsd(rs,ss,is_num):
    try:
        if is_num:
            lo=min(rs.min(),ss.min()); hi=max(rs.max(),ss.max())
            if lo>=hi: return 0.0
            p,_=np.histogram(rs.dropna(),30,(lo,hi))
            q,_=np.histogram(ss.dropna(),30,(lo,hi))
        else:
            cats=pd.Categorical(pd.concat([rs,ss]).dropna()).categories
            p=rs.value_counts().reindex(cats,fill_value=0).values
            q=ss.value_counts().reindex(cats,fill_value=0).values
        p=p.astype(float)+1e-10; q=q.astype(float)+1e-10
        return float(jensenshannon(p/p.sum(),q/q.sum()))
    except: return float("nan")

def _cd(r,s):
    rv=r.dropna().values.astype(float); sv=s.dropna().values.astype(float)
    if len(rv)<2 or len(sv)<2: return float("nan")
    p=math.sqrt((np.var(rv,ddof=1)+np.var(sv,ddof=1))/2)
    return 0.0 if p==0 else abs(float(np.mean(rv)-np.mean(sv))/p)

def _corr_diff(Xr,Xs,proc,mc=60):
    sel=list(proc.num_dims)
    for s,e,K in proc.cat_groups:
        sel.append(s+int(Xr[:,s:e].var(0).argmax()))
    if len(sel)>mc:
        vv=Xr[:,sel].var(0); sel=[sel[i] for i in np.argsort(vv)[::-1][:mc]]
    idx=np.array(sel,dtype=int)
    Xrr=np.nan_to_num(Xr[:,idx].astype(float))
    Xss=np.nan_to_num(Xs[:,idx].astype(float))
    keep=(Xrr.var(0)>1e-8)&(Xss.var(0)>1e-8)
    Xrr=Xrr[:,keep]; Xss=Xss[:,keep]
    if Xrr.shape[1]<2: return float("nan")
    rc=np.nan_to_num(np.corrcoef(Xrr.T)); sc=np.nan_to_num(np.corrcoef(Xss.T))
    mask=~np.eye(Xrr.shape[1],dtype=bool)
    return float(np.abs(rc[mask]-sc[mask]).mean())

def evaluate_all(name, real_df, syn_df, Xr, Xs, proc):
    real_df=_strip(real_df); syn_df=_strip(syn_df)
    num_r={c for c in real_df.columns if pd.api.types.is_numeric_dtype(real_df[c])}

    # JSD
    jsds=[_jsd(real_df[c],syn_df[c],c in num_r)
          for c in real_df.columns if c in syn_df.columns]
    jsd_mean=float(np.nanmean(jsds)) if jsds else float("nan")

    # Cohen's d
    ns=[c for c in num_r if c in syn_df.columns
        and pd.api.types.is_numeric_dtype(syn_df[c])]
    cd=[_cd(pd.to_numeric(real_df[c],errors="coerce"),
            pd.to_numeric(syn_df[c],errors="coerce")) for c in ns]
    avg_cd=float(np.nanmean(cd)) if cd else float("nan")

    stat_sim=float((real_df[ns].apply(lambda c:pd.to_numeric(c,errors="coerce")).mean()-
                    syn_df[ns].apply(lambda c:pd.to_numeric(c,errors="coerce")).mean()
                   ).abs().mean()) if ns else float("nan")

    corr_diff=_corr_diff(Xr,Xs,proc,80)

    # ML classifier TRTS
    ml_acc=ml_f1=ml_auc=float("nan")
    try:
        vo=np.argsort(Xr.var(0))[::-1]
        tgt=None
        for c in vo[:50]:
            med=float(np.median(Xr[:,c]))
            yr=(Xr[:,c]>med).astype(int); ys=(Xs[:,c]>med).astype(int)
            if 0.10<yr.mean()<0.90 and 0.10<ys.mean()<0.90:
                tgt=int(c); y_r=yr; y_s=ys; break
        if tgt is not None:
            fd=[i for i in range(Xr.shape[1]) if i!=tgt]
            ii=np.random.choice(len(Xs),min(2000,len(Xs)),replace=False)
            clf=RandomForestClassifier(200,max_depth=12,random_state=SEED,
                                        n_jobs=-1,class_weight="balanced")
            clf.fit(Xs[ii][:,fd],y_s[ii])
            yp=clf.predict(Xr[:,fd])
            ml_acc=float(accuracy_score(y_r,yp))
            ml_f1 =float(f1_score(y_r,yp,zero_division=0,average="macro"))
            pr=clf.predict_proba(Xr[:,fd]); cl=list(clf.classes_)
            if 1 in cl and len(np.unique(y_r))==2:
                ml_auc=float(roc_auc_score(y_r,pr[:,cl.index(1)]))
    except Exception as e: print(f"    [eval] ML: {e}")

    # R² top-5 numeric
    reg_r2=float("nan"); reg_tgt="N/A"
    try:
        nd=proc.num_dims
        if nd:
            rv=Xr[:,nd].var(0); sv=Xs[:,nd].var(0)
            order=np.argsort(rv)[::-1]
            qual=[lo for lo in order if rv[lo]>0 and sv[lo]>=0.05*rv[lo]]
            top=qual[:5] if qual else list(order[:5])
            r2s=[]; names=[]
            for lo in top:
                bd=nd[lo]; names.append(proc.col_name(bd))
                fd=[i for i in range(Xr.shape[1]) if i!=bd]
                ii=np.random.choice(len(Xs),min(3000,len(Xs)),replace=False)
                rfr=RandomForestRegressor(150,max_depth=12,random_state=SEED,n_jobs=-1)
                rfr.fit(Xs[ii][:,fd],Xs[ii,bd])
                r2s.append(float(r2_score(Xr[:,bd],rfr.predict(Xr[:,fd]))))
            reg_r2=float(np.mean(r2s)); reg_tgt="+".join(names[:3])
            print("    [eval] R²: "+", ".join(f"{n}={v:.4f}" for n,v in zip(names,r2s)))
    except Exception as e: print(f"    [eval] R²: {e}")

    # Privacy DCR
    dcr=priv=float("nan")
    try:
        from sklearn.metrics.pairwise import euclidean_distances
        ri=np.random.choice(len(Xr),min(500,len(Xr)),replace=False)
        si=np.random.choice(len(Xs),min(500,len(Xs)),replace=False)
        ds=euclidean_distances(Xs[si],Xr[ri]).min(1).mean()
        ri2=np.random.choice(len(Xr),min(500,len(Xr)),replace=False)
        ri3=np.random.choice(len(Xr),min(500,len(Xr)),replace=False)
        Drr=euclidean_distances(Xr[ri2],Xr[ri3]); np.fill_diagonal(Drr,np.inf)
        dr=Drr.min(1).mean(); dcr=float(ds); priv=float(ds/dr) if dr>0 else float("nan")
    except Exception as e: print(f"    [eval] privacy: {e}")

    def _r(v,d=4): return round(v,d) if isinstance(v,float) and not math.isnan(v) else "nan"
    return {"method":name,"jsd_mean":_r(jsd_mean,5),"stat_sim":_r(stat_sim,4),
            "avg_cohens_d":_r(avg_cd,4),"corr_diff":_r(corr_diff,4),
            "ml_accuracy":_r(ml_acc,4),"ml_f1_macro":_r(ml_f1,4),"ml_auc":_r(ml_auc,4),
            "reg_r2":_r(reg_r2,4),"privacy_dcr":_r(dcr,4),"privacy_ratio":_r(priv,4),
            "reg_target":reg_tgt}

# ==============================================================================
# 23.  PLOTS (unchanged)
# ==============================================================================

_PS={"figure.facecolor":"white","axes.facecolor":"white","text.color":"#111",
     "grid.color":"#ccc","grid.linestyle":"--","grid.alpha":0.6,
     "font.family":"DejaVu Sans","font.size":10,
     "axes.spines.top":False,"axes.spines.right":False}
_CR="#1565C0"; _CS="#E63946"
_PAL=["#E63946","#1565C0","#2E7D32","#F57C00","#6A1B9A","#F9A825","#00838F"]

def _sty():
    if _HAS_PLOT: plt.rcParams.update(_PS)

def plot_loss(lh,name):
    if not _HAS_PLOT or not lh: return
    _sty(); fig,ax=plt.subplots(figsize=(10,4),facecolor="white")
    ep=np.arange(1,len(lh)+1); ax.plot(ep,lh,color="#90CAF9",lw=1.2,alpha=0.8)
    if len(lh)>=10:
        w=max(len(lh)//20,5); sm=np.convolve(lh,np.ones(w)/w,"valid")
        ax.plot(ep[w-1:],sm,color=_CS,lw=2.5,label=f"Smooth(w={w})")
    ax.set_title(f"Loss – {name}",fontweight="bold"); ax.legend(); ax.grid(True)
    plt.tight_layout(); fp=PLOT_DIR/f"loss_{name}.png"
    plt.savefig(fp,dpi=150,bbox_inches="tight",facecolor="white"); plt.close()

def plot_corr(Xr,Xs,names,name,mc=25):
    if not _HAS_PLOT: return
    _sty(); nc=min(mc,Xr.shape[1])
    vi=np.argsort(Xr.var(0))[::-1][:nc]
    ln=[names[i] if i<len(names) else str(i) for i in vi]
    Xrr=np.nan_to_num(Xr[:,vi].astype(float)); Xss=np.nan_to_num(Xs[:,vi].astype(float))
    rc=pd.DataFrame(Xrr,columns=ln).corr(); sc=pd.DataFrame(Xss,columns=ln).corr()
    fig,axes=plt.subplots(1,3,figsize=(20,7),facecolor="white")
    kw=dict(cmap="coolwarm",vmin=-1,vmax=1,xticklabels=False,yticklabels=False,linewidths=0.2)
    sns.heatmap(rc,ax=axes[0],**kw); axes[0].set_title("Real",color=_CR,fontweight="bold")
    sns.heatmap(sc,ax=axes[1],**kw); axes[1].set_title("Synthetic",color=_CS,fontweight="bold")
    sns.heatmap((rc-sc).abs(),ax=axes[2],cmap="YlOrRd",vmin=0,vmax=1,xticklabels=False,yticklabels=False,linewidths=0.2)
    axes[2].set_title("|Diff|",color="#2E7D32",fontweight="bold")
    plt.tight_layout(); fp=PLOT_DIR/f"corr_{name}.png"
    plt.savefig(fp,dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"    [plot] corr_{name}.png")

def plot_comparison(results):
    if not _HAS_PLOT or not results: return
    _sty()
    mets=["jsd_mean","avg_cohens_d","corr_diff","ml_accuracy","ml_f1_macro","ml_auc","reg_r2","privacy_ratio"]
    lmap={"jsd_mean":"JSD↓","avg_cohens_d":"Avg|d|↓","corr_diff":"CorrDiff↓",
          "ml_accuracy":"Acc↑","ml_f1_macro":"F1↑","ml_auc":"AUC↑","reg_r2":"R²↑","privacy_ratio":"PrivR↑"}
    nm=len(mets); nr=len(results); x=np.arange(nm); w=0.8/max(nr,1)
    fig,ax=plt.subplots(figsize=(16,6),facecolor="white")
    for i,row in enumerate(results):
        vals=[float(row.get(m,0)) if row.get(m,"nan")!="nan" else 0.0 for m in mets]
        ax.bar(x+i*w-(nr-1)*w/2,vals,w*0.9,label=row["method"],color=_PAL[i%len(_PAL)],alpha=0.88,edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([lmap[m] for m in mets],rotation=20,ha="right")
    ax.legend(loc="upper right"); ax.set_title("Comparison",fontweight="bold"); ax.grid(axis="y")
    plt.tight_layout(); plt.savefig(PLOT_DIR/"comparison.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"    [plot] comparison.png")

def plot_tsne(Xr,Xs,name,n=1500):
    if not(_HAS_PLOT and _HAS_MANIFOLD): return
    _sty()
    from sklearn.preprocessing import StandardScaler
    nr=min(n,len(Xr)); ns=min(n,len(Xs))
    ir=np.random.choice(len(Xr),nr,replace=False); is_=np.random.choice(len(Xs),ns,replace=False)
    Xa=np.vstack([Xr[ir],Xs[is_]]); lb=np.array(["Real"]*nr+["Syn"]*ns)
    Xsc=StandardScaler().fit_transform(Xa)
    emb=TSNE(2,random_state=SEED,perplexity=min(30,max(nr//3,5)),init="pca",learning_rate="auto").fit_transform(Xsc)
    fig,ax=plt.subplots(figsize=(8,7),facecolor="white")
    for l,c in [("Real",_CR),("Syn",_CS)]:
        m=lb==l; ax.scatter(emb[m,0],emb[m,1],s=10,alpha=0.5,color=c,label=l,linewidths=0)
    ax.set_title(f"t-SNE – {name}",fontweight="bold"); ax.legend(markerscale=3); ax.grid(alpha=0.4)
    plt.tight_layout(); plt.savefig(PLOT_DIR/f"tsne_{name}.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"    [plot] tsne_{name}.png")

def plot_pca(Xr,Xs,name,n=2000):
    if not(_HAS_PLOT and _HAS_MANIFOLD): return
    _sty()
    from sklearn.preprocessing import StandardScaler
    nr=min(n,len(Xr)); ns=min(n,len(Xs))
    ir=np.random.choice(len(Xr),nr,replace=False); is_=np.random.choice(len(Xs),ns,replace=False)
    Xa=StandardScaler().fit_transform(np.vstack([Xr[ir],Xs[is_]]))
    pca=PCA(2,random_state=SEED); emb=pca.fit_transform(Xa)
    ev=pca.explained_variance_ratio_; lb=np.array(["Real"]*nr+["Syn"]*ns)
    fig,axes=plt.subplots(1,2,figsize=(14,6),facecolor="white")
    for l,c in [("Real",_CR),("Syn",_CS)]:
        m=lb==l; axes[0].scatter(emb[m,0],emb[m,1],s=8,alpha=0.5,color=c,label=l,linewidths=0)
    axes[0].set_title(f"PCA – {name}",fontweight="bold")
    axes[0].set_xlabel(f"PC1({ev[0]*100:.1f}%)"); axes[0].legend(markerscale=3); axes[0].grid(alpha=0.4)
    for l,c in [("Real",_CR),("Syn",_CS)]:
        m=lb==l; v=emb[m,0]; xs=np.linspace(v.min(),v.max(),300)
        try:
            kde=scipy_stats.gaussian_kde(v)
            axes[1].fill_between(xs,kde(xs),alpha=0.18,color=c); axes[1].plot(xs,kde(xs),color=c,lw=2.5,label=l)
        except: pass
    axes[1].set_title(f"PC1 Density – {name}",fontweight="bold"); axes[1].legend(); axes[1].grid(alpha=0.4)
    plt.tight_layout(); plt.savefig(PLOT_DIR/f"pca_{name}.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"    [plot] pca_{name}.png")

# ==============================================================================
# 24.  RUNNER (updated dispatch)
# ==============================================================================

class _W:
    def __init__(self,fn): self._fn=fn
    def sample(self,n): return self._fn(n)

def run_one(name, train_fn, X_gpu, proc, real_df, n_synth, pure_syn, X_real_enc):
    print(f"\n{'─'*64}\n  ▶  {name}  │  {n_synth} rows  │  {X_gpu.shape[1]} dims\n{'─'*64}")
    t0=time.time(); lh=[]
    res=train_fn(X_gpu,lh)
    obj=res if hasattr(res,"sample") else _W(res)
    Xs=obj.sample(n_synth).astype(np.float32)

    # --- NEW: constraint-violation rate BEFORE any clipping (isolates PGD effect) ---
    if proc.num_dims:
        raw_num = Xs[:, proc.num_dims]
        oob_frac = float(((raw_num < 0.0) | (raw_num > 1.0)).mean())
    else:
        oob_frac = 0.0

    # Enforce one-hot validity
    for s,e,K in proc.cat_groups:
        chunk=Xs[:,s:e]; idx=chunk.argmax(1)
        oh=np.zeros_like(chunk); oh[np.arange(len(idx)),idx]=1.0; Xs[:,s:e]=oh
    # Clip numeric
    for d in proc.num_dims: Xs[:,d]=np.clip(Xs[:,d],0.0,1.0)

    # Numeric calibration
    Xs = _calibrate_numeric(Xs, X_real_enc, proc.num_dims)

    syn_df=proc.inverse_transform(Xs)
    fp=OUT/f"{name}_synthetic_n{n_synth}.csv"
    syn_df.to_csv(fp,index=False,encoding="utf-8-sig"); print(f"    → {fp}")

    if not pure_syn:
        pd.concat([real_df.assign(_source="real"),syn_df.assign(_source="synthetic")]
                  ).to_csv(OUT/f"{name}_combined_n{n_synth}.csv",index=False,encoding="utf-8-sig")

    m=evaluate_all(name,real_df,syn_df,X_real_enc,Xs,proc)
    m["time_s"]=round(time.time()-t0,1); m["n_synth"]=n_synth
    m["numeric_oob_frac"]=round(oob_frac,5)   # <-- NEW
    print(f"    JSD={m['jsd_mean']}  AvgD={m['avg_cohens_d']}  CorrDiff={m['corr_diff']}\n"
          f"    Acc={m['ml_accuracy']}  F1={m['ml_f1_macro']}  AUC={m['ml_auc']}  R²={m['reg_r2']}\n"
          f"    DCR={m['privacy_dcr']}  PR={m['privacy_ratio']}  OOB={m['numeric_oob_frac']}  ({m['time_s']}s)")

    # Plots
    plot_loss(lh, name)
    enc_names=[]
    for sp,(s,e) in zip(proc.specs,proc.slices):
        if sp.kind=="num": enc_names.append(sp.name)
        else: enc_names+=[f"{sp.name}_{k}" for k in range(e-s)]
    plot_corr(X_real_enc,Xs,enc_names,name)
    plot_tsne(X_real_enc,Xs,name)
    plot_pca(X_real_enc,Xs,name)
    return m

# ==============================================================================
# 25.  ABLATION COMPONENT TABLE (markdown + LaTeX)
# ==============================================================================

def print_ablation_component_table():
    rows=[]
    stage_label={"one_stage":"1-stage","two_stage":"2-stage","three_stage":"3-stage"}
    for name,cfg in HIDETAB_ABLATION_CONFIGS.items():
        rows.append({
            "Variant": name,
            "TC Penalty": "✓" if cfg["lambda_tc"]>0 else "✗",
            "Disentangled Latent": "✓",
            "Hierarchy": stage_label[cfg["stage_mode"]],
            "FiLM Cond.": ("✓" if cfg["use_film"] else "✗") if cfg["stage_mode"]!="one_stage" else "–",
        })
    dfc=pd.DataFrame(rows)
    print(dfc.to_string(index=False))
    dfc.to_csv(OUT/"ablation_components.csv", index=False, encoding="utf-8-sig")
    return dfc


def latex_ablation_component_table():
    """Returns a ready-to-paste LaTeX table string."""
    stage_label={"one_stage":"1","two_stage":"2","three_stage":"3"}
    lines=[r"\begin{table}[t]",
           r"\centering",
           r"\caption{HiDe-Tab ablation: components used by each variant.}",
           r"\label{tab:ablation-components}",
           r"\begin{tabular}{lcccc}",
           r"\toprule",
           r"Variant & TC Penalty & Disent.\ Latent & \#Stages & FiLM Cond. \\",
           r"\midrule"]
    for name,cfg in HIDETAB_ABLATION_CONFIGS.items():
        tc    = r"\checkmark" if cfg["lambda_tc"]>0 else "--"
        film  = (r"\checkmark" if cfg["use_film"] else "--") if cfg["stage_mode"]!="one_stage" else "n/a"
        nstage = stage_label[cfg["stage_mode"]]
        label  = name.replace("_", r"\_")
        lines.append(f"{label} & {tc} & \\checkmark & {nstage} & {film} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

# ==============================================================================
# 26.  DISPATCH + MAIN
# ==============================================================================

def _dispatch(proc):
    D = {
        "CTGAN":    lambda X, lh: _W(train_ctgan(X, proc, loss_history=lh)),
        "Diffusion":lambda X, lh: _W(train_diffusion(X, proc, loss_history=lh)),
        "HiDe_Tab": lambda X, lh: _W(train_hidetab(X, proc, loss_history=lh)),
        "TabDiff":  lambda X, lh: _W(train_tabdiff(X, proc, loss_history=lh)),
        "TabSyn":   lambda X, lh: _W(train_tabsyn(X, proc, loss_history=lh)),
    }
    # auto-register ablation variants (no use_pgd)
    for vname, cfg in HIDETAB_ABLATION_CONFIGS.items():
        D[vname] = (lambda X, lh, vn=vname, c=cfg:
                    _W(train_hidetab_variant(X, proc, name=vn, loss_history=lh, **c)))
    return D

def main():
    sep="═"*68
    print(f"\n{sep}")
    print("  Kult2012 Synthetic Data  v9 + HiDe-Tab + TabDiff + TabSyn")
    dev_str=f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type=="cuda" else ""
    print(f"  Device: {DEVICE}{dev_str}")
    print(f"  Epochs  VAE={EPOCHS_VAE}  pretrain={EPOCHS_VAE_PRETRAIN}  "
          f"GAN={EPOCHS_GAN}  Diff={EPOCHS_DIFF}  FT={EPOCHS_FINETUNE}")
    print(f"  Batch={BATCH_SIZE}  LR={LR}  LATENT={LATENT_DIM}  "
          f"Stride={TRANSFORMER_STRIDE}  num_w={NUM_LOSS_WEIGHT}")
    print(f"  Methods: {', '.join(METHODS)}")
    print(f"{sep}\n")

    raw=load_sav(DATA_PATH); df=preprocess_df(raw); print()
    proc=DataProcessor(); X_np=proc.fit_transform(df)
    print(f"[info] Encoded {X_np.shape}  cat={len(proc.cat_groups)} num={len(proc.num_dims)}\n")
    X=torch.tensor(X_np,dtype=torch.float32)
    print(f"[info] Tensor {tuple(X.shape)}  ({X.element_size()*X.nelement()/1024**2:.1f} MB)\n")

    D=_dispatch(proc); results=[]
    for name in METHODS:
        if name not in D: print(f"[skip] {name}"); continue
        m=run_one(name,D[name],X,proc,df,N_SYNTH,PURE_SYNTHETIC,X_np)
        results.append(m)

    if results:
        plot_comparison(results)
        df_s=pd.DataFrame(results); fp=OUT/"summary_metrics.csv"
        df_s.to_csv(fp,index=False,encoding="utf-8-sig")
        cols = ["method", "jsd_mean", "avg_cohens_d", "corr_diff", "ml_accuracy",
                "ml_f1_macro", "ml_auc", "reg_r2", "privacy_dcr", "privacy_ratio",
                "numeric_oob_frac", "time_s"]
        print(f"\n{sep}\n  SUMMARY\n{sep}")
        print(df_s[cols].to_string(index=False))
        print(f"\n  CSV  → {fp.resolve()}\n  Plots→ {PLOT_DIR.resolve()}")

if __name__=="__main__":
    main()