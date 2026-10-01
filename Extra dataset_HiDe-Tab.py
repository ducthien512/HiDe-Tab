"""
HiDe-Tab on Four Public Sociological Datasets
Methods : CTGAN | Diffusion | HiDe-Tab | TabDiff | TabSyn
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATHS = {
    "adult":  "data/adult/adult.data",
    "german": "data/german/german.data",
    "ess":    "data/ESS10/ESS10.csv",
    "gss":    "data/GSS2022/GSS2022.dta",
}
DATASETS_TO_RUN = ["adult", "german", "ess", "gss"]
METHODS = ["CTGAN", "Diffusion", "HiDe_Tab", "TabDiff", "TabSyn"]

FIXED_THR = {"gss": 0.5}          # datasets whose _target_src is already an indicator

N_SYNTH = 20000
N_SEEDS = 3
OUTPUT_DIR = "outputs_extra"
CALIBRATE_NUMERIC = False          # if True, applied to ALL methods equally

EPOCHS_VAE_PRETRAIN = 80
EPOCHS_GAN = 150
EPOCHS_DIFF = 200
EPOCHS_FINETUNE = 20
MIN_STEPS = 2500                   # minimum gradient steps (helps small datasets)

BATCH_SIZE = 256
LR = 1e-4
LATENT_DIM = 64
HIDDEN_DIM = 256
T_STEPS = 200
LAT_S, LAT_V, LAT_C = 16, 32, 16

NUM_LOSS_WEIGHT = 80.0
NUM_FT_EXTRA = 6.0
CTGAN_NUM_WEIGHT = 5.0

CAT_TEMP = 1.0
EMA_DECAY = 0.999
USE_DDIM = True
DDIM_STEPS = 50
DDIM_ETA = 0.5

MAX_CAT_LEVELS = 15
MAX_MISS_FRAC = 0.50
N_REF_RR = 100                     # n_ref for RR_p5
RANDOM_SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
import os, sys, math, time, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except ImportError:
    sys.exit("[ERROR] PyTorch not found. Install with: pip install torch")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[device] {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == "cuda": torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_adult(path):
    cols = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation",
            "relationship","race","sex","capital_gain","capital_loss","hours_per_week",
            "native_country","income"]
    df = pd.read_csv(path, header=None, names=cols, na_values=[" ?","?"], skipinitialspace=True)
    df = df.drop(columns=["fnlwgt"])
    df["income_binary"] = df["income"].str.strip().str.replace(".", "", regex=False).eq(">50K").astype(int)
    df = df.drop(columns=["income"])
    print(f"  [Adult] {df.shape}  pos={df['income_binary'].mean():.2%}")
    return df, "income_binary"

def load_german(path):
    df = pd.read_csv(path, header=None, sep=r"\s+")
    df.columns = [f"A{i}" for i in range(1, df.shape[1]+1)]
    df["credit_risk"] = (df["A21"] - 1).astype(int)
    df = df.drop(columns=["A21"])
    print(f"  [German] {df.shape}  pos={df['credit_risk'].mean():.2%}")
    return df, "credit_risk"

def _mask_missing(df, spec, default=(77, 88, 99)):
    for c in df.columns:
        df[c] = df[c].replace(list(spec.get(c, default)), np.nan)
    return df

def load_ess(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    KEEP = ["gndr","agea","eduyrs","isco08","emplrel","wrkhome","wrklong","health","hlthhmp","happy",
            "stflife","stfeco","stfgov","stfdem","trstprl","trstlgl","trstplc","trstplt","trstep","trstun",
            "imtcjob","imueclt","imwbcnt","rlgdgr","pray","freehms","hmsfmlsh","hmsacld","euftf","lrscale",
            "polintr","vote","psppsgva","actrolga","cptppola","wrkprty","netuse","inprdsc","sclmeet",
            "socnhh","maritalb","hhmmb","domicil","ctzcntr","brncntr","hinctnta","hhinctnta"]
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df = _mask_missing(df, {"agea": [999], "isco08": [66666,77777,88888,99999]})
    drop_const = [c for c in df.columns
                  if len(df[c].dropna()) and float(df[c].value_counts(normalize=True).iloc[0]) > 0.90]
    if drop_const:
        df = df.drop(columns=drop_const); print(f"  [ESS] Dropped {len(drop_const)} near-constant columns")
    src = "stflife" if "stflife" in df.columns else "happy"
    df["_target_src"] = pd.to_numeric(df[src], errors="coerce")
    df = df.drop(columns=[c for c in ["stflife","happy"] if c in df.columns])
    df = df[df["_target_src"].notna()].reset_index(drop=True)      # no NaN -> class 0
    df["stflife_binary"] = np.nan
    print(f"  [ESS R10] {df.shape}  target from '{src}' (median split on train)")
    return df, "stflife_binary"

def load_gss(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_stata(path.replace(".csv", ".dta"), convert_categoricals=False)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    KEEP = ["age","sex","race","educ","degree","marital","childs","sibs","income","rincome","class_","satjob",
            "happy","life","health","hapmar","polviews","partyid","trust","helpful","fair","prayer","reliten",
            "relig","attend","bible","conarmy","conbus","conclerg","concourt","coneduc","confed","confinan",
            "conjudge","conlegis","conmedic","conpress","consci","conlabor","hrs1","wrkstat","occ10","prestg10",
            "indus10","unemp","fechld","fefam","fepresch","fepol","natenvir","natheal","nateduc","natrace",
            "natdrug","natcrime","cappun","gunlaw","grass","pornlaw","letdie1","suicide1","abany"]
    df = df[[c for c in KEEP if c in df.columns]].copy()
    for mv in [0, 98, 99, 998, 999, 9998, 9999]:
        df = df.replace(mv, np.nan)
    happy = pd.to_numeric(df["happy"], errors="coerce")
    df = df[happy.notna()].copy(); happy = happy[happy.notna()]
    df["_target_src"] = (happy == 1).astype(float)                  # 1 = very happy
    df = df.drop(columns=["happy"]).reset_index(drop=True)
    df["happy_binary"] = np.nan
    print(f"  [GSS 2022] {df.shape}  target: very happy vs pretty/not too happy")
    return df, "happy_binary"

def derive_binary_target(df_train, df_test, target_col, fixed_thr=None):
    """Threshold from TRAIN only; applied unchanged to test."""
    if "_target_src" not in df_train.columns:
        return df_train, df_test, 0.5
    st = pd.to_numeric(df_train["_target_src"], errors="coerce")
    se = pd.to_numeric(df_test["_target_src"], errors="coerce")
    if fixed_thr is not None:
        thr = fixed_thr
    else:
        cands = [float(st.median())] + st.dropna().quantile([1/3, 2/3]).tolist()
        thr = cands[0]; best = abs(float((st >= thr).mean()) - 0.5)
        for c in cands:
            pos = float((st >= c).mean()); imb = abs(pos - 0.5)
            if 0.15 <= pos <= 0.85 and imb < best: best, thr = imb, c
    df_train = df_train.copy(); df_test = df_test.copy()
    df_train[target_col] = (st >= thr).astype(int)
    df_test[target_col] = (se >= thr).astype(int)
    return df_train.drop(columns=["_target_src"]), df_test.drop(columns=["_target_src"]), thr

# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSOR  (target column is now INCLUDED as a 2-level categorical)
# ══════════════════════════════════════════════════════════════════════════════
class ColumnSpec:
    def __init__(self, name):
        self.name = name; self.kind = None; self.is_int = False; self.fill = None
        self.scaler = MinMaxScaler(clip=True); self.cats = None; self.fill_cat = None; self.n = 1

    def fit(self, raw):
        c = pd.to_numeric(raw, errors="coerce")
        if c.notna().mean() >= 0.5 and raw.nunique(dropna=True) > MAX_CAT_LEVELS:
            self.kind = "num"; valid = c.dropna()
            self.is_int = len(valid) > 0 and bool((valid == valid.round()).all())
            self.fill = float(valid.median()) if len(valid) else 0.0
            self.scaler.fit(c.fillna(self.fill).values.reshape(-1, 1)); self.n = 1
        else:
            self.kind = "cat"
            s = raw.astype(str).where(raw.notna(), other=np.nan)
            top = list(s.dropna().value_counts().index[:MAX_CAT_LEVELS])
            self.fill_cat = top[0] if top else "__UNK__"; self.cats = top; self.n = len(top)
        return self

    def transform(self, raw):
        if self.kind == "num":
            v = pd.to_numeric(raw, errors="coerce").fillna(self.fill)
            return self.scaler.transform(v.values.reshape(-1, 1)).ravel().astype(np.float32)
        s = raw.astype(str).where(raw.notna(), np.nan).fillna(self.fill_cat).values
        mat = np.zeros((len(s), len(self.cats)), np.float32)
        ci = {c: i for i, c in enumerate(self.cats)}
        for r, v in enumerate(s): mat[r, ci.get(v, ci.get(self.fill_cat, 0))] = 1.0
        return mat

    def inverse_transform(self, arr):
        if self.kind == "num":
            u = self.scaler.inverse_transform(np.asarray(arr).reshape(-1, 1)).ravel()
            return np.round(u).astype(np.int64) if self.is_int else u
        mat = np.asarray(arr)
        if mat.ndim == 1: mat = mat.reshape(-1, 1)
        return np.array(self.cats)[np.clip(mat.argmax(1), 0, len(self.cats) - 1)]

class DataProcessor:
    def __init__(self, target_col=None):
        self.target_col = target_col; self.columns = []; self.specs = []; self.slices = []; self._n = 0

    def fit(self, df):
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        self.columns = [c for c in df.columns if not c.startswith("_")
                        and (c == self.target_col or bool(df[c].notna().mean() > (1 - MAX_MISS_FRAC)))]
        self.specs = [ColumnSpec(c).fit(df[c]) for c in self.columns]
        i = 0
        for sp in self.specs: self.slices.append((i, i + sp.n)); i += sp.n
        self._n = i
        self.t_idx = self.columns.index(self.target_col)
        print(f"  [proc] {len(self.columns)} cols (incl. target) -> {i} dims  "
              f"(num={sum(s.kind=='num' for s in self.specs)}, cat={sum(s.kind=='cat' for s in self.specs)})")
        return self

    def transform(self, df):
        parts = []
        for sp, c in zip(self.specs, self.columns):
            o = sp.transform(df[c]); parts.append(o.reshape(-1, 1) if o.ndim == 1 else o)
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df): return self.fit(df).transform(df)

    def inverse_transform(self, X):
        out = {}
        for sp, (s, e) in zip(self.specs, self.slices):
            ch = X[:, s:e]; out[sp.name] = sp.inverse_transform(ch.ravel() if sp.kind == "num" else ch)
        return pd.DataFrame(out)

    def split_target(self, X):
        """-> (features without target one-hot, binary label from target one-hot)."""
        sp = self.specs[self.t_idx]; s, e = self.slices[self.t_idx]
        lab = np.array(sp.cats)[X[:, s:e].argmax(1)]
        return np.delete(X, np.arange(s, e), axis=1), (lab == "1").astype(int)

    @property
    def num_dims(self): return [s for sp, (s, e) in zip(self.specs, self.slices) if sp.kind == "num"]
    @property
    def cat_groups(self): return [(s, e, e - s) for sp, (s, e) in zip(self.specs, self.slices) if sp.kind == "cat"]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
class GPULoader:
    """Whole dataset on GPU, batches by index -> no DataLoader overhead."""
    def __init__(self, *tensors, batch=None, shuffle=True):
        self.T = [t.to(DEVICE) for t in tensors]; self.n = len(self.T[0])
        self.b = batch or BATCH_SIZE; self.shuffle = shuffle
    def __len__(self):
        full, rem = divmod(self.n, self.b); return max(1, full + (1 if rem >= 8 else 0))
    def __iter__(self):
        idx = torch.randperm(self.n, device=DEVICE) if self.shuffle else torch.arange(self.n, device=DEVICE)
        for s in range(0, self.n, self.b):
            j = idx[s:s + self.b]
            if len(j) < 8 and s > 0: break
            yield tuple(t[j] for t in self.T)

def make_loader(X, batch=None, shuffle=True): return GPULoader(X, batch=batch, shuffle=shuffle)
def gpu(t): return t.to(DEVICE, non_blocking=True)

def eff_epochs(n, epochs, min_steps=None):
    ms = MIN_STEPS if min_steps is None else min_steps
    return max(epochs, int(math.ceil(ms / max(1, n // BATCH_SIZE))))

class EMA:
    """GPU-resident EMA with warm-up (fixes decay=0.9999 on short runs)."""
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay; self.n = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        self.n += 1; d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else: self.shadow[k].copy_(v)
    def apply(self, model): model.load_state_dict(self.shadow)

def composite_loss(recon, target, cat_groups, num_dims, num_w=NUM_LOSS_WEIGHT):
    loss = torch.tensor(0.0, device=recon.device); n = 0
    for s, e, K in cat_groups:
        loss = loss + F.cross_entropy(recon[:, s:e], target[:, s:e].argmax(1), label_smoothing=0.05); n += 1
    if num_dims:
        idx = torch.tensor(num_dims, device=recon.device, dtype=torch.long)
        loss = loss + num_w * F.mse_loss(recon[:, idx], target[:, idx]); n += 1
    return loss / max(n, 1)

def _cyc_beta(ep, epochs, cycles=6, bmax=0.05):
    cl = max(epochs // cycles, 1); return bmax * min((ep % cl) / cl * 2.0, 1.0)

def _norm_latent(Z):
    Zm = Z.mean(0, keepdim=True); Zs = Z.std(0, keepdim=True).clamp(1e-3); return (Z - Zm) / Zs, Zm, Zs

def _calibrate_numeric(X_syn, X_real, num_dims):
    out = X_syn.copy()
    for d in num_dims:
        s_std = max(float(out[:, d].std()), 1e-6)
        out[:, d] = np.clip((out[:, d] - out[:, d].mean()) / s_std * X_real[:, d].std() + X_real[:, d].mean(), 0, 1)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# NETWORK BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
class _Res(nn.Module):
    def __init__(self, d):
        super().__init__(); self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d))
    def forward(self, x): return x + self.ff(self.ln(x))

class SinPE(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        h = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / max(h - 1, 1))
        a = t[:, None].float() * f[None]; return torch.cat([a.sin(), a.cos()], -1)

class FiLMBlock(nn.Module):
    def __init__(self, dim, td):
        super().__init__(); self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Dropout(0.05), nn.Linear(dim*2, dim))
        self.film = nn.Sequential(nn.Linear(td, dim*2), nn.SiLU(), nn.Linear(dim*2, dim*2))
    def forward(self, x, te):
        sc, sh = self.film(te).chunk(2, -1); return x + self.ff(self.norm(x) * (1 + sc) + sh)

class MultiHeadDecoder(nn.Module):
    def __init__(self, lat, hid, cat_groups, num_dims, total_dims):
        super().__init__()
        self.cat_groups = cat_groups; self.num_dims = num_dims; self.total = total_dims
        self.trunk = nn.Sequential(nn.Linear(lat, hid), nn.LayerNorm(hid), nn.GELU(),
                                   nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
                                   nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU())
        self.cat_heads = nn.ModuleList([nn.Linear(hid, K) for _, _, K in cat_groups])
        if num_dims:
            nh = max(hid // 2, 64)
            self.num_trunk = nn.Sequential(nn.Linear(lat, nh), nn.LayerNorm(nh), nn.GELU(),
                                           nn.Linear(nh, nh), nn.LayerNorm(nh), nn.GELU(),
                                           nn.Linear(nh, nh), nn.GELU())
            self.num_head = nn.Linear(nh, len(num_dims))
        else:
            self.num_trunk = self.num_head = None

    def _num(self, z, out):
        if self.num_head is not None:
            idx = torch.tensor(self.num_dims, device=z.device, dtype=torch.long)
            out[:, idx] = torch.sigmoid(self.num_head(self.num_trunk(z)))
        return out

    def forward(self, z):
        h = self.trunk(z); out = torch.zeros(z.size(0), self.total, device=z.device)
        for head, (s, e, K) in zip(self.cat_heads, self.cat_groups): out[:, s:e] = head(h)
        return self._num(z, out)

    def forward_soft(self, z, tau=0.5):            # differentiable one-hot for GANs
        h = self.trunk(z); out = torch.zeros(z.size(0), self.total, device=z.device)
        for head, (s, e, K) in zip(self.cat_heads, self.cat_groups):
            out[:, s:e] = F.gumbel_softmax(head(h), tau=tau, hard=True)
        return self._num(z, out)

    @torch.no_grad()
    def sample(self, z, temp=CAT_TEMP):
        h = self.trunk(z); out = torch.zeros(z.size(0), self.total, device=z.device)
        for head, (s, e, K) in zip(self.cat_heads, self.cat_groups):
            p = F.softmax(head(h) / max(temp, 1e-6), -1)
            out[:, s:e] = F.one_hot(torch.multinomial(p, 1).squeeze(-1), K).float()
        return self._num(z, out)

# ══════════════════════════════════════════════════════════════════════════════
# DDPM / DDIM
# ══════════════════════════════════════════════════════════════════════════════
class DDPMScheduler:
    def __init__(self, T=T_STEPS, s=0.008):
        self.T = T; steps = torch.arange(T + 1, dtype=torch.float32)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2; f = f / f[0]
        self.betas = (1.0 - f[1:] / f[:-1]).clamp(1e-5, 0.999).to(DEVICE); self.alpha_bar = f[1:].to(DEVICE)

    def q_sample(self, x0, t):
        ab = self.alpha_bar[t].view(-1, 1); eps = torch.randn_like(x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond=None):
        x = torch.randn(shape, device=DEVICE)
        call = (lambda x, tb: model(x, tb)) if cond is None else (lambda x, tb: model(x, tb, cond))
        if USE_DDIM:
            ts = sorted(set(torch.linspace(self.T - 1, 0, min(DDIM_STEPS, self.T)).round().long().tolist()), reverse=True)
            for k, i in enumerate(ts):
                eps = call(x, torch.full((shape[0],), i, device=DEVICE, dtype=torch.long))
                ab = self.alpha_bar[i]
                x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-5, 5)
                if k + 1 == len(ts): return x0
                ab_p = self.alpha_bar[ts[k + 1]]
                sig = DDIM_ETA * ((1 - ab_p) / (1 - ab) * (1 - ab / ab_p)).clamp(min=0).sqrt()
                x = ab_p.sqrt() * x0 + (1 - ab_p - sig**2).clamp(min=0).sqrt() * eps + sig * torch.randn_like(x)
            return x
        for i in reversed(range(self.T)):
            eps = call(x, torch.full((shape[0],), i, device=DEVICE, dtype=torch.long))
            ab = self.alpha_bar[i]; ab_p = self.alpha_bar[i-1] if i > 0 else torch.tensor(1.0, device=DEVICE)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-5, 5)
            if i == 0: return x0
            b = self.betas[i]
            mean = (ab_p.sqrt() * b / (1 - ab)) * x0 + (ab.sqrt() * (1 - ab_p) / (1 - ab)) * x
            x = mean + (b * (1 - ab_p) / (1 - ab)).clamp(1e-20).sqrt() * torch.randn_like(x)
        return x

class DiffNet(nn.Module):
    """FiLM-MLP denoiser; optional conditioning vector (concatenated to input)."""
    def __init__(self, lat_d=LATENT_DIM, hid=HIDDEN_DIM, n_blocks=6, cond_dim=0):
        super().__init__()
        self.te = nn.Sequential(SinPE(hid), nn.Linear(hid, hid*2), nn.SiLU(), nn.Linear(hid*2, hid))
        self.xp = nn.Sequential(nn.Linear(lat_d + cond_dim, hid), nn.LayerNorm(hid), nn.SiLU())
        self.blocks = nn.ModuleList([FiLMBlock(hid, hid) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, lat_d))
    def forward(self, x, t, cond=None):
        if cond is not None: x = torch.cat([x, cond], -1)
        te = self.te(t); h = self.xp(x)
        for b in self.blocks: h = b(h, te)
        return self.out(h)

def _train_diff_on_latent(Zn, lat_d, epochs, loss_history, tag, lr_mult=2.0, n_blocks=6):
    model = DiffNet(lat_d, HIDDEN_DIM, n_blocks).to(DEVICE); ema = EMA(model); sch = DDPMScheduler(T_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR * lr_mult, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, max(epochs // 4, 20), eta_min=LR * 0.05)
    ldr = make_loader(Zn)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for (zb,) in ldr:
            t = torch.randint(0, T_STEPS, (zb.size(0),), device=DEVICE); zt, noise = sch.q_sample(zb, t)
            loss = F.mse_loss(model(zt, t), noise)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model); tot += loss.item()
        lrs.step()
        if loss_history is not None: loss_history.append(tot / len(ldr))
        if ep % 100 == 0 or ep == epochs: print(f"    [{tag}] ep {ep:4d}/{epochs}  loss={tot/len(ldr):.5f}")
    ema.apply(model); model.eval(); return model, sch

# ══════════════════════════════════════════════════════════════════════════════
# VAEs
# ══════════════════════════════════════════════════════════════════════════════
class CVAEEnc(nn.Module):
    def __init__(self, in_d, hid, lat):
        super().__init__(); mid = min(hid * 2, 1024)
        self.proj = nn.Sequential(nn.Linear(in_d, mid), nn.LayerNorm(mid), nn.GELU(),
                                  nn.Linear(mid, hid), nn.LayerNorm(hid), nn.GELU())
        self.res = nn.Sequential(*[_Res(hid) for _ in range(4)]); self.norm = nn.LayerNorm(hid)
        self.mu = nn.Linear(hid, lat); self.lv = nn.Linear(hid, lat)
    def forward(self, x):
        h = self.norm(self.res(self.proj(x))); return self.mu(h), self.lv(h).clamp(-4, 4)

class CVAE(nn.Module):
    def __init__(self, in_d, cat_groups, num_dims, hid=HIDDEN_DIM, lat=LATENT_DIM):
        super().__init__(); self.enc = CVAEEnc(in_d, hid, lat)
        self.dec = MultiHeadDecoder(lat, hid, cat_groups, num_dims, in_d)
    def forward(self, x):
        mu, lv = self.enc(x); return self.dec(mu + torch.exp(0.5 * lv) * torch.randn_like(mu)), mu, lv
    @torch.no_grad()
    def encode_mu(self, x): return self.enc(x)[0]

@torch.no_grad()
def _encode_all(fn, X, batch=1024):
    return torch.cat([fn(gpu(X[s:s + batch])).cpu() for s in range(0, len(X), batch)])

def _train_vae_core(model, X, proc, epochs, tag="VAE", loss_history=None):
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, max(epochs // 6, 10), eta_min=LR * 0.05)
    ldr = make_loader(X); best, bst = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; bkl = _cyc_beta(ep, epochs)
        for (xb,) in ldr:
            r, mu, lv = model(xb); rl = composite_loss(r, xb, proc.cat_groups, proc.num_dims)
            loss = rl + bkl * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += rl.item()
        sch.step(); l = tot / len(ldr)
        if l < best: best = l; bst = {k: v.clone() for k, v in model.state_dict().items()}
        if loss_history is not None: loss_history.append(l)
        if ep % 40 == 0 or ep == epochs: print(f"    [{tag}] ep {ep:4d}/{epochs}  recon={l:.5f}")
    model.load_state_dict(bst); return model

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 – CTGAN
# ══════════════════════════════════════════════════════════════════════════════
class _CTGANGen(nn.Module):
    def __init__(self, nd, hid, cat_groups, num_dims, total_dims):
        super().__init__()
        mk = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU())
        self.fc0, self.fc1, self.fc2, self.fc3 = mk(nd, hid), mk(hid, hid), mk(hid, hid), mk(hid, hid)
        self.skip = nn.Linear(nd, hid); self.head = MultiHeadDecoder(hid, hid, cat_groups, num_dims, total_dims)
    def _trunk(self, z):
        h0 = self.fc0(z); h1 = self.fc1(h0); h2 = self.fc2(h1 + h0); return self.fc3(h2) + self.skip(z)
    def forward(self, z): return self.head.forward_soft(self._trunk(z))
    def sample(self, z): return self.head.sample(self._trunk(z))

class _CTGANDisc(nn.Module):
    def __init__(self, id_, hid):
        super().__init__(); SN = nn.utils.spectral_norm
        self.net = nn.Sequential(SN(nn.Linear(id_, hid)), nn.LeakyReLU(0.2), nn.Dropout(0.25),
                                 SN(nn.Linear(hid, hid)), nn.LeakyReLU(0.2), nn.Dropout(0.25),
                                 SN(nn.Linear(hid, hid // 2)), nn.LeakyReLU(0.2), SN(nn.Linear(hid // 2, 1)))
    def forward(self, x): return self.net(x)

def _gp(D, real, fake, lam=10.0):
    a = torch.rand(real.size(0), 1, device=real.device)
    ip = (a * real + (1 - a) * fake).requires_grad_(True)
    g = torch.autograd.grad(D(ip), ip, grad_outputs=torch.ones(ip.size(0), 1, device=ip.device), create_graph=True)[0]
    return lam * ((g.norm(2, dim=1) - 1) ** 2).mean()

def train_ctgan(X, proc, epochs=EPOCHS_GAN, nd=LATENT_DIM, loss_history=None):
    epochs = eff_epochs(len(X), epochs, MIN_STEPS // 2); id_ = X.shape[1]
    G = _CTGANGen(nd, HIDDEN_DIM, proc.cat_groups, proc.num_dims, id_).to(DEVICE)
    D = _CTGANDisc(id_, HIDDEN_DIM).to(DEVICE)
    oG = torch.optim.Adam(G.parameters(), lr=LR * 0.5, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))
    sG = torch.optim.lr_scheduler.CosineAnnealingLR(oG, epochs); sD = torch.optim.lr_scheduler.CosineAnnealingLR(oD, epochs)
    ldr = make_loader(X)
    idx = torch.tensor(proc.num_dims, device=DEVICE, dtype=torch.long) if proc.num_dims else None
    for ep in range(1, epochs + 1):
        dL = gL = 0.0
        for (xb,) in ldr:
            bs = xb.size(0)
            for _ in range(5):
                fk = G(torch.randn(bs, nd, device=DEVICE)).detach()
                ld = -D(xb).mean() + D(fk).mean() + _gp(D, xb, fk); oD.zero_grad(); ld.backward(); oD.step()
            for _ in range(2):
                fk = G(torch.randn(bs, nd, device=DEVICE)); lg = -D(fk).mean()
                if idx is not None:
                    lg = lg + CTGAN_NUM_WEIGHT * (F.mse_loss(fk[:, idx].mean(0), xb[:, idx].mean(0)) +
                                                  F.mse_loss(fk[:, idx].std(0).clamp(1e-4), xb[:, idx].std(0).clamp(1e-4)))
                oG.zero_grad(); lg.backward(); oG.step()
            dL += ld.item(); gL += lg.item()
        sG.step(); sD.step()
        if loss_history is not None: loss_history.append(gL / len(ldr))
        if ep % 100 == 0 or ep == epochs: print(f"    [CTGAN] ep {ep:4d}/{epochs}  D={dL/len(ldr):.4f}  G={gL/len(ldr):.4f}")
    G.eval()
    def sampler(n):
        with torch.no_grad():
            return np.vstack([G.sample(torch.randn(min(2048, n - s), nd, device=DEVICE)).cpu().numpy()
                              for s in range(0, n, 2048)])
    return sampler

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 – Diffusion
# ══════════════════════════════════════════════════════════════════════════════
def train_diffusion(X, proc, epochs=EPOCHS_DIFF, loss_history=None):
    n = len(X)
    vae = CVAE(X.shape[1], proc.cat_groups, proc.num_dims).to(DEVICE)
    _train_vae_core(vae, X, proc, eff_epochs(n, EPOCHS_VAE_PRETRAIN, MIN_STEPS // 2), tag="Diff/VAE")
    vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]
    Zn, Zm, Zs = _norm_latent(_encode_all(vae.encode_mu, X))
    model, sch = _train_diff_on_latent(Zn, Zn.shape[1], eff_epochs(n, epochs), loss_history, "Diffusion")
    def sampler(n_):
        Zm_d, Zs_d = Zm.to(DEVICE), Zs.to(DEVICE); parts = []
        for s in range(0, n_, 512):
            z = sch.p_sample_loop(model, (min(512, n_ - s), Zn.shape[1])) * Zs_d + Zm_d
            parts.append(vae.dec.sample(z, CAT_TEMP).cpu().numpy())
        return np.vstack(parts).astype(np.float32)
    return sampler

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 – HiDe-Tab  (Hierarchical Disentangled latent diffusion)
#   Stage 1 : disentangled VAE (z_s | z_v | z_c), shared residual encoder, scale-invariant TC
#   Stage 2 : macro DDPM on z_sc=[z_s,z_c]  (A)  +  micro DDPM on z_v | z_sc  (B), trained jointly
#   Stage 3 : short decoder fine-tune (encoder frozen)
# ══════════════════════════════════════════════════════════════════════════════
class DisentangledCVAE(nn.Module):
    def __init__(self, in_d, cat_groups, num_dims, hid=HIDDEN_DIM, lat_s=LAT_S, lat_v=LAT_V, lat_c=LAT_C):
        super().__init__(); mid = min(hid * 2, 1024)
        self.sizes = [lat_s, lat_v, lat_c]; L = sum(self.sizes); self.lat_s = lat_s
        self.trunk = nn.Sequential(nn.Linear(in_d, mid), nn.LayerNorm(mid), nn.GELU(),
                                   nn.Linear(mid, hid), nn.LayerNorm(hid), nn.GELU(),
                                   *[_Res(hid) for _ in range(3)], nn.LayerNorm(hid))
        self.mu = nn.Linear(hid, L); self.lv = nn.Linear(hid, L)
        self.dec = MultiHeadDecoder(L, hid, cat_groups, num_dims, in_d)
    def encode_dist(self, x):
        h = self.trunk(x); return self.mu(h), self.lv(h).clamp(-4, 4)
    def forward(self, x):
        mu, lv = self.encode_dist(x); z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.dec(z), mu, lv, z
    def groups(self, z): return z.split(self.sizes, dim=1)

def tc_corr_loss(groups):
    """Scale-invariant cross-group correlation penalty."""
    zs = [(g - g.mean(0, keepdim=True)) / (g.std(0, keepdim=True) + 1e-3) for g in groups]
    B = zs[0].size(0); loss = zs[0].new_tensor(0.0); k = 0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            loss = loss + (zs[i].T @ zs[j] / (B - 1)).pow(2).mean(); k += 1
    return loss / max(k, 1)

def train_hidetab(X, proc, seed=RANDOM_SEED, loss_history=None, lam_tc=0.05):
    set_seed(seed); n = len(X); ldr = make_loader(X)
    # ---- Stage 1
    ep_v = eff_epochs(n, EPOCHS_VAE_PRETRAIN, MIN_STEPS // 2)
    print(f"  [HiDe-Tab] Stage 1 — Disentangled VAE ({ep_v} ep)")
    vae = DisentangledCVAE(X.shape[1], proc.cat_groups, proc.num_dims).to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=LR * 3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, max(ep_v // 6, 10), eta_min=LR * 0.05)
    best, bst = float("inf"), None
    for ep in range(1, ep_v + 1):
        vae.train(); tr = tk = tt = 0.0; bkl = _cyc_beta(ep, ep_v)
        for (xb,) in ldr:
            rec, mu, lv, z = vae(xb); rl = composite_loss(rec, xb, proc.cat_groups, proc.num_dims)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()); tc = tc_corr_loss(vae.groups(z))
            loss = rl + bkl * kl + lam_tc * tc
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step()
            tr += rl.item(); tk += kl.item(); tt += tc.item()
        sch.step(); l = tr / len(ldr)
        if l < best: best = l; bst = {k: v.clone() for k, v in vae.state_dict().items()}
        if loss_history is not None: loss_history.append(l)
        if ep % 40 == 0 or ep == ep_v: print(f"    ep {ep:4d}/{ep_v}  recon={l:.4f}  KL={tk/len(ldr):.3f}  TC={tt/len(ldr):.4f}")
    vae.load_state_dict(bst); vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]

    # ---- Encode
    def enc_mu(xb):
        mu, _ = vae.encode_dist(xb); s, v, c = vae.groups(mu); return torch.cat([s, c, v], 1)
    Z = _encode_all(enc_mu, X); sc_dim = LAT_S + LAT_C; v_dim = LAT_V
    Z_sc, Z_v = Z[:, :sc_dim], Z[:, sc_dim:]
    Zn_sc, Zsc_m, Zsc_s = _norm_latent(Z_sc); Zn_v, Zv_m, Zv_s = _norm_latent(Z_v)
    print(f"    Latent dims: z_sc={sc_dim}, z_v={v_dim}")

    # ---- Stage 2: A and B trained jointly in one pass (B teacher-forced on real z_sc)
    ep_d = eff_epochs(n, EPOCHS_DIFF); print(f"  [HiDe-Tab] Stage A+B — joint latent diffusion ({ep_d} ep)")
    mA = DiffNet(sc_dim, HIDDEN_DIM, 4).to(DEVICE); mB = DiffNet(v_dim, HIDDEN_DIM, 4, cond_dim=sc_dim).to(DEVICE)
    eA, eB = EMA(mA), EMA(mB); ds = DDPMScheduler(T_STEPS)
    opt = torch.optim.AdamW(list(mA.parameters()) + list(mB.parameters()), lr=LR * 2, weight_decay=1e-4)
    lrs = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, max(ep_d // 4, 20), eta_min=LR * 0.05)
    ldr2 = GPULoader(Zn_sc, Zn_v)
    for ep in range(1, ep_d + 1):
        mA.train(); mB.train(); tot = 0.0
        for zsc, zv in ldr2:
            ta = torch.randint(0, T_STEPS, (zsc.size(0),), device=DEVICE); tb = torch.randint(0, T_STEPS, (zsc.size(0),), device=DEVICE)
            za, na = ds.q_sample(zsc, ta); zb, nb = ds.q_sample(zv, tb)
            loss = F.mse_loss(mA(za, ta), na) + F.mse_loss(mB(zb, tb, zsc), nb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(mA.parameters()) + list(mB.parameters()), 1.0)
            opt.step(); eA.update(mA); eB.update(mB); tot += loss.item()
        lrs.step()
        if loss_history is not None: loss_history.append(tot / len(ldr2))
        if ep % 100 == 0 or ep == ep_d: print(f"    ep {ep:4d}/{ep_d}  loss(A+B)={tot/len(ldr2):.5f}")
    eA.apply(mA); eB.apply(mB); mA.eval(); mB.eval()

    # ---- Stage 3: decoder fine-tune
    ep_f = eff_epochs(n, EPOCHS_FINETUNE, MIN_STEPS // 4); print(f"  [HiDe-Tab] Stage C — decoder fine-tune ({ep_f} ep)")
    [p.requires_grad_(True) for p in vae.dec.parameters()]
    od = torch.optim.AdamW(vae.dec.parameters(), lr=LR * 0.8, weight_decay=1e-5)
    sd = torch.optim.lr_scheduler.CosineAnnealingLR(od, ep_f, eta_min=LR * 0.01)
    for ep in range(1, ep_f + 1):
        vae.dec.train(); tot = 0.0
        for (xb,) in ldr:
            with torch.no_grad():
                mu, lv = vae.encode_dist(xb); z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            loss = composite_loss(vae.dec(z), xb, proc.cat_groups, proc.num_dims, num_w=NUM_LOSS_WEIGHT * NUM_FT_EXTRA)
            od.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(vae.dec.parameters(), 1.0); od.step(); tot += loss.item()
        sd.step()
        if ep % 20 == 0 or ep == ep_f: print(f"    ep {ep:3d}/{ep_f}  dec={tot/len(ldr):.5f}")
    vae.eval()

    def sampler(n_):
        a, b, c, d = Zsc_m.to(DEVICE), Zsc_s.to(DEVICE), Zv_m.to(DEVICE), Zv_s.to(DEVICE); parts = []
        for s0 in range(0, n_, 512):
            cur = min(512, n_ - s0)
            zsc_n = ds.p_sample_loop(mA, (cur, sc_dim)); zv_n = ds.p_sample_loop(mB, (cur, v_dim), cond=zsc_n)
            zsc = zsc_n * b + a; zv = zv_n * d + c
            z = torch.cat([zsc[:, :LAT_S], zv, zsc[:, LAT_S:]], 1)      # order: s | v | c
            parts.append(vae.dec.sample(z, CAT_TEMP).cpu().numpy())
        return np.vstack(parts).astype(np.float32)
    return sampler

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4 – TabDiff (AE baseline)     METHOD 5 – TabSyn (VAE + latent diffusion)
# ══════════════════════════════════════════════════════════════════════════════
class TabDiff(nn.Module):
    def __init__(self, in_d, cat_groups, num_dims, hid=HIDDEN_DIM, lat=LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d, hid), nn.LayerNorm(hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(), nn.Linear(hid, lat))
        self.dec = MultiHeadDecoder(lat, hid, cat_groups, num_dims, in_d); self.lat = lat
    def forward(self, x): return self.dec(self.enc(x))

def train_tabdiff(X, proc, epochs=EPOCHS_DIFF, loss_history=None):
    epochs = eff_epochs(len(X), epochs); print(f"  [TabDiff] Training ({epochs} ep)")
    model = TabDiff(X.shape[1], proc.cat_groups, proc.num_dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5); ldr = make_loader(X)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for (xb,) in ldr:
            loss = composite_loss(model(xb), xb, proc.cat_groups, proc.num_dims)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        sch.step()
        if loss_history is not None: loss_history.append(tot / len(ldr))
        if ep % 100 == 0 or ep == epochs: print(f"    ep {ep:4d}/{epochs}  loss={tot/len(ldr):.5f}")
    model.eval()
    def sampler(n):
        with torch.no_grad():
            return model.dec.sample(torch.randn(n, model.lat, device=DEVICE), CAT_TEMP).cpu().numpy()
    return sampler

def train_tabsyn(X, proc, epochs=EPOCHS_DIFF, loss_history=None):
    n = len(X); ep_v = eff_epochs(n, EPOCHS_VAE_PRETRAIN, MIN_STEPS // 2); print(f"  [TabSyn] Stage 1 — VAE ({ep_v} ep)")
    vae = CVAE(X.shape[1], proc.cat_groups, proc.num_dims).to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep_v, eta_min=1e-5); ldr = make_loader(X); best, bst = float("inf"), None
    for ep in range(1, ep_v + 1):
        vae.train(); tot = 0.0
        for (xb,) in ldr:
            r, mu, lv = vae(xb)
            loss = composite_loss(r, xb, proc.cat_groups, proc.num_dims) + 0.1 * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step(); tot += loss.item()
        sch.step(); l = tot / len(ldr)
        if l < best: best = l; bst = {k: v.clone() for k, v in vae.state_dict().items()}
        if ep % 40 == 0 or ep == ep_v: print(f"    [TabSyn/VAE] ep {ep:4d}/{ep_v}  loss={l:.5f}")
    vae.load_state_dict(bst); vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]
    Zn, Zm, Zs = _norm_latent(_encode_all(vae.encode_mu, X))
    print("  [TabSyn] Stage 2 — Diffusion on latent")
    dm, dsch = _train_diff_on_latent(Zn, Zn.shape[1], eff_epochs(n, epochs), loss_history, "TabSyn")
    def sampler(n_):
        Zm_d, Zs_d = Zm.to(DEVICE), Zs.to(DEVICE); parts = []
        for s in range(0, n_, 512):
            z = dsch.p_sample_loop(dm, (min(512, n_ - s), Zn.shape[1])) * Zs_d + Zm_d
            parts.append(vae.dec.sample(z, CAT_TEMP).cpu().numpy())
        return np.vstack(parts).astype(np.float32)
    return sampler

class _SamplerWrapper:
    def __init__(self, fn): self.fn = fn
    def sample(self, n): return self.fn(n)

DISPATCH = {
    "CTGAN":     lambda X, p, lh, seed: _SamplerWrapper(train_ctgan(X, p, loss_history=lh)),
    "Diffusion": lambda X, p, lh, seed: _SamplerWrapper(train_diffusion(X, p, loss_history=lh)),
    "HiDe_Tab":  lambda X, p, lh, seed: _SamplerWrapper(train_hidetab(X, p, seed=seed, loss_history=lh)),
    "TabDiff":   lambda X, p, lh, seed: _SamplerWrapper(train_tabdiff(X, p, loss_history=lh)),
    "TabSyn":    lambda X, p, lh, seed: _SamplerWrapper(train_tabsyn(X, p, loss_history=lh)),
}

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
NAN = float("nan")

def _tstr(Ftr, ytr, Fte, yte, seed):
    """Train RF on (Ftr,ytr), evaluate on (Fte,yte). -> acc, macro-F1, AUC"""
    if len(np.unique(ytr)) < 2: return NAN, NAN, NAN
    clf = RandomForestClassifier(200, max_depth=12, random_state=seed, n_jobs=-1, class_weight="balanced").fit(Ftr, ytr)
    yp = clf.predict(Fte)
    auc = (float(roc_auc_score(yte, clf.predict_proba(Fte)[:, list(clf.classes_).index(1)]))
           if len(np.unique(yte)) == 2 else NAN)
    return float(accuracy_score(yte, yp)), float(f1_score(yte, yp, average="macro", zero_division=0)), auc

def _min_dist(A, B, ref_idx=None, chunk=1024):
    """min Euclidean distance from each row of A to rows of B (GPU, chunked).
       If ref_idx is given, A[i] is B[ref_idx[i]] and self-distance is excluded."""
    At = torch.tensor(A, device=DEVICE); Bt = torch.tensor(B, device=DEVICE); out = []
    for i in range(0, len(At), chunk):
        d = torch.cdist(At[i:i + chunk], Bt, compute_mode="donot_use_mm_for_euclid_dist")
        if ref_idx is not None:
            r = torch.arange(d.size(0), device=DEVICE); d[r, torch.as_tensor(ref_idx[i:i + chunk], device=DEVICE)] = float("inf")
        out.append(d.min(1).values.cpu())
    return torch.cat(out).numpy()

def evaluate_all(name, df_test, syn_df, Xr, Xs, proc, X_train_enc, real_only, seed):
    rng = np.random.default_rng(seed); res = {"method": name}; tcol = proc.target_col

    # ── JSD (marginal fidelity) and Avg|Cohen's d| (numeric cols); column type from processor
    jsds, cds = [], []
    for sp in proc.specs:
        c = sp.name
        if c == tcol: continue
        try:
            if sp.kind == "num":
                r = pd.to_numeric(df_test[c], errors="coerce").dropna().values.astype(float)
                s = pd.to_numeric(syn_df[c], errors="coerce").dropna().values.astype(float)
                nP, nQ = len(r), len(s)
                if nP > 1 and nQ > 1:
                    sp_ = math.sqrt(((nP - 1) * r.var(ddof=1) + (nQ - 1) * s.var(ddof=1)) / (nP + nQ - 2))
                    cds.append(0.0 if sp_ == 0 else abs(r.mean() - s.mean()) / sp_)
                lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
                if lo >= hi: continue
                p, _ = np.histogram(r, 30, (lo, hi)); q, _ = np.histogram(s, 30, (lo, hi))
            else:
                r = df_test[c][df_test[c].notna()].astype(str); s = syn_df[c].astype(str)
                cats = sorted(set(r) | set(s))
                p = r.value_counts().reindex(cats, fill_value=0).values; q = s.value_counts().reindex(cats, fill_value=0).values
            p = p.astype(float) + 1e-10; q = q.astype(float) + 1e-10
            jsds.append(float(jensenshannon(p / p.sum(), q / q.sum())))
        except Exception:
            pass
    res["jsd_mean"] = float(np.mean(jsds)) if jsds else NAN
    res["avg_cohens_d"] = float(np.mean(cds)) if cds else NAN          # Eq. (1)

    # ── Corr = (1/m^2) sum_ij |R_ij - S_ij|  over all encoded dims with real variance > 0     Eq. (2)
    keep = Xr.var(0) > 1e-8
    if keep.sum() >= 2:
        rc = np.corrcoef(Xr[:, keep].T)
        sc = np.nan_to_num(np.corrcoef(np.nan_to_num(Xs[:, keep]).T)); np.fill_diagonal(sc, 1.0)
        res["corr_diff"] = float(np.abs(rc - sc).mean())
    else:
        res["corr_diff"] = NAN

    # ── TSTR with the synthetic target (jointly generated)
    Fr, yr = proc.split_target(Xr); Fs, ys = proc.split_target(Xs)
    res["pos_rate_syn"] = float(ys.mean())
    res["ml_accuracy"], res["ml_f1_macro"], res["ml_auc"] = _tstr(Fs, ys, Fr, yr, seed)
    res["f1_real_only"], res["auc_real_only"] = real_only

    # ── Privacy: DCR_p5, RR_p5, PR   Eq. (3)-(5)
    dcr = _min_dist(Xs.astype(np.float32), X_train_enc)
    res["dcr_p5"] = float(np.percentile(dcr, 5))
    ref = rng.choice(len(X_train_enc), min(N_REF_RR, len(X_train_enc)), replace=False)
    rr = _min_dist(X_train_enc[ref], X_train_enc, ref_idx=ref)
    res["rr_p5"] = float(np.percentile(rr, 5))
    res["privacy_ratio"] = res["dcr_p5"] / (res["rr_p5"] + 1e-8)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}

def run_method(method, X_train_t, proc, X_train_enc, X_test_enc, df_test, n_synth, out_dir, seed, real_only):
    set_seed(seed)
    print(f"\n  {'─'*56}\n  ▶  {method}  │  seed={seed}  │  {n_synth} rows\n  {'─'*56}")
    t0 = time.time(); lh = []; obj = DISPATCH[method](X_train_t, proc, lh, seed); t_train = time.time() - t0
    t0 = time.time(); Xs = obj.sample(n_synth).astype(np.float32); t_sample = time.time() - t0
    for s, e, K in proc.cat_groups:
        idx = Xs[:, s:e].argmax(1); oh = np.zeros_like(Xs[:, s:e]); oh[np.arange(len(idx)), idx] = 1.0; Xs[:, s:e] = oh
    for d in proc.num_dims: Xs[:, d] = np.clip(Xs[:, d], 0.0, 1.0)
    if CALIBRATE_NUMERIC: Xs = _calibrate_numeric(Xs, X_train_enc, proc.num_dims)   # all methods
    syn_df = proc.inverse_transform(Xs)
    syn_df.to_csv(out_dir / f"{method}_synthetic_seed{seed}.csv", index=False, encoding="utf-8-sig")
    m = evaluate_all(method, df_test, syn_df, X_test_enc, Xs, proc, X_train_enc, real_only, seed)
    m["train_time_s"] = round(t_train, 1); m["sample_time_s"] = round(t_sample, 1); m["seed"] = seed
    print(f"  JSD={m['jsd_mean']}  Avg|d|={m['avg_cohens_d']}  Corr={m['corr_diff']}  F1={m['ml_f1_macro']}  "
          f"AUC={m['ml_auc']}  DCR5={m['dcr_p5']}  RR5={m['rr_p5']}  PR={m['privacy_ratio']}  pos_syn={m['pos_rate_syn']:.2f}")
    print(f"  [Real-only] F1={m['f1_real_only']}  AUC={m['auc_real_only']}   Train={m['train_time_s']}s  Sample={m['sample_time_s']}s")
    del obj
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return m

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
LOADERS = {"adult": load_adult, "german": load_german, "ess": load_ess, "gss": load_gss}
SEEDS = [RANDOM_SEED + i for i in range(N_SEEDS)]
METRIC_KEYS = ["jsd_mean", "avg_cohens_d", "corr_diff", "ml_accuracy", "ml_f1_macro", "ml_auc", "f1_real_only",
               "auc_real_only", "dcr_p5", "rr_p5", "privacy_ratio", "pos_rate_syn", "train_time_s", "sample_time_s"]
all_dataset_results = {}

for DATASET in DATASETS_TO_RUN:
    path = DATA_PATHS[DATASET]
    if not os.path.exists(path):
        alt = path.replace(".dta", ".csv")
        if os.path.exists(alt): path = alt
        else:
            print(f"\n[SKIP] {DATASET}: file not found at '{path}'"); continue

    print(f"\n{'═'*64}\n  DATASET: {DATASET.upper()}\n  Methods: {', '.join(METHODS)}  |  Seeds: {SEEDS}\n{'═'*64}")
    out_dir = Path(OUTPUT_DIR) / DATASET; out_dir.mkdir(parents=True, exist_ok=True)
    df_full, target_col = LOADERS[DATASET](path)

    if "_target_src" in df_full.columns:
        ts = df_full["_target_src"]
        strat = ts if ts.nunique() <= 5 else pd.qcut(ts, 4, labels=False, duplicates="drop")
    else:
        strat = df_full[target_col]
    df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=RANDOM_SEED, stratify=strat)
    df_train, df_test, thr = derive_binary_target(df_train, df_test, target_col, FIXED_THR.get(DATASET))
    print(f"\n  Train={len(df_train)}  Test={len(df_test)}  target='{target_col}'  thr={thr:.2f}  "
          f"pos(train)={df_train[target_col].mean():.2%}  pos(test)={df_test[target_col].mean():.2%}")

    proc = DataProcessor(target_col=target_col)
    X_train_enc = proc.fit_transform(df_train); X_test_enc = proc.transform(df_test)
    X_train_t = torch.tensor(X_train_enc, dtype=torch.float32)
    Ft, yt = proc.split_target(X_train_enc); Fr, yr = proc.split_target(X_test_enc)
    print(f"  Encoded dims: {X_train_enc.shape[1]}  (cat_groups={len(proc.cat_groups)}, num_dims={len(proc.num_dims)})")

    res_by_method = {m: [] for m in METHODS}
    for seed in SEEDS:
        print(f"\n{'·'*64}\n  SEED {seed}\n{'·'*64}")
        _, f1_ro, auc_ro = _tstr(Ft, yt, Fr, yr, seed)
        for method in METHODS:
            try:
                res_by_method[method].append(run_method(method, X_train_t, proc, X_train_enc, X_test_enc,
                                                        df_test, N_SYNTH, out_dir, seed, (round(f1_ro, 4), round(auc_ro, 4))))
            except Exception as e:
                import traceback; print(f"  [ERROR] {method} seed={seed}: {e}"); traceback.print_exc()

    print(f"\n{'═'*64}\n  SUMMARY  {DATASET.upper()}  ({N_SEEDS} seeds)\n{'═'*64}")
    agg_rows = []
    for method, runs in res_by_method.items():
        if not runs: continue
        row = {"dataset": DATASET, "method": method, "n_seeds": len(runs)}
        for k in METRIC_KEYS:
            v = [r[k] for r in runs if isinstance(r.get(k), (int, float)) and not math.isnan(r[k])]
            row[f"{k}_mean"] = round(float(np.mean(v)), 4) if v else NAN
            row[f"{k}_std"] = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0
        agg_rows.append(row)
        print(f"\n  {method}:")
        for k in ["jsd_mean", "avg_cohens_d", "corr_diff", "ml_f1_macro", "ml_auc", "dcr_p5", "rr_p5", "privacy_ratio", "train_time_s"]:
            print(f"    {k:16s} {row[f'{k}_mean']:.4f} ± {row[f'{k}_std']:.4f}")
        print(f"    real-only F1={row['f1_real_only_mean']}  AUC={row['auc_real_only_mean']}")
    pd.DataFrame(agg_rows).to_csv(out_dir / f"summary_{DATASET}_{N_SEEDS}seeds.csv", index=False, encoding="utf-8-sig")
    all_dataset_results[DATASET] = agg_rows

print(f"\n{'═'*64}\n  CROSS-DATASET SUMMARY\n{'═'*64}")
rows = [r for rs in all_dataset_results.values() for r in rs]
if rows:
    df_all = pd.DataFrame(rows); sp = Path(OUTPUT_DIR) / "all_datasets_summary.csv"
    df_all.to_csv(sp, index=False, encoding="utf-8-sig")
    print(df_all[["dataset", "method", "jsd_mean_mean", "avg_cohens_d_mean", "corr_diff_mean", "ml_f1_macro_mean",
                  "ml_auc_mean", "privacy_ratio_mean", "train_time_s_mean"]].to_string(index=False))
    print(f"\n  -> {sp.resolve()}")
print(f"\n{'═'*64}\n  DONE\n{'═'*64}")
