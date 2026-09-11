# ============================================================
# PIAAC SYNTHETIC DATA GENERATOR
# Supports: 500 / 1000 feature modes
# Methods: CTGAN, Diffusion, HiDe-Tab, TabDiff, TabSyn
# ============================================================

# ============================================================
# CONFIGURATION — Edit these settings
# ============================================================

# Choose targer sample size: 5000 / 10000 / 20000
TARGET_SIZE    = 5000
DATA_PATH = "PIAAC PUF dataset\\Cycle 1\\prgczep1.csv" # Cycle 1
#DATA_PATH = "PIAAC PUF dataset\\Cycle 2\\prgczep2.csv" # Cycle 2

# Choose feature mode: 500 / 1000
FEATURE_MODE   = 500

# PIAAC cycle: 1 or 2
CYCLE = 1

assert CYCLE in (1, 2), "CYCLE must be 1 or 2"

# If True  → generate TARGET_SIZE purely synthetic rows
# If False → generate (TARGET_SIZE - n_original) rows,
#            then combine with real data to reach TARGET_SIZE
PURE_SYNTHETIC = True

# Missing value threshold: drop columns with more than this % missing
# 1.00 → keep ALL requested features, impute everything
MISSING_THRESHOLD = 1.00

# Random seed — set once here, propagated to NumPy, TF, sklearn everywhere
RANDOM_SEED    = 42
# Number of repeated evaluation runs (for mean ± std reporting in paper)
N_EVAL_RUNS    = 5

# ============================================================
# PyTorch‑based method hyperparameters (HiDe‑Tab, TabDiff, TabSyn)
# ============================================================
PT_EPOCHS_VAE   = 100
PT_EPOCHS_DIFF  = 100
PT_BATCH_SIZE   = 256
PT_LR           = 2e-4
PT_LATENT_DIM   = 16
PT_HIDDEN_DIM   = 256
PT_T_STEPS      = 150
PT_LAT_S        = 24
PT_LAT_V        = 48
PT_LAT_C        = 24
PT_CAT_TEMP            = 0.8
PT_NUM_LOSS_WEIGHT     = 80.0
PT_NUM_FT_EXTRA        = 6.0

# TabSyn specific
PT_TABSYN_LATENT       = 64

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, mean_squared_error,
                              r2_score)
from sklearn.ensemble import (RandomForestClassifier,
                               RandomForestRegressor)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
import json
import re as _re

# PyTorch imports for new methods
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("Warning: PyTorch not installed. HiDe-Tab, TabDiff, TabSyn will be skipped.")

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _pt_gpu(t):
    return t.to(DEVICE)

# ── Reproducibility: fix all random seeds ────────────────────────────────────
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
import random as _random
_random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
if _HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

for d in ["outputs/comparison", "outputs/plots",
          "outputs/data", "outputs/logs"]:
    os.makedirs(d, exist_ok=True)

# Log environment for reproducibility
import platform, sys as _sys
_env_info = {
    'python':   _sys.version,
    'platform': platform.platform(),
    'tf':       tf.__version__,
    'sklearn':  __import__('sklearn').__version__,
    'pandas':   pd.__version__,
    'numpy':    np.__version__,
    'scipy':    __import__('scipy').__version__,
}
with open('outputs/logs/environment.txt', 'w') as _ef:
    for k, v in _env_info.items():
        _ef.write(f"{k}: {v}\n")

# ============================================================
# FEATURE DEFINITIONS  (automatic — no manual lists needed)
# ============================================================
# Features are selected automatically from the CSV after loading:
#   1. Exclude always-excluded identifier / weight / routing columns.
#   2. Drop columns with > MISSING_THRESHOLD missing values.
#   3. Drop all-NaN columns.
#   4. If the CSV has more than FEATURE_MODE usable columns, keep the
#      FEATURE_MODE columns with the lowest missing-value rate (ties
#      broken by column order).  Set FEATURE_MODE = 0 to keep all.
# ── placeholder so the rest of the file can reference it ────────────────
BLOCK_DEMOGRAPHICS = []  # kept for backward compat; not used by auto-select
# ── All manual BLOCK_* lists removed — features selected automatically ──────
# (see preprocessing section below)

EXCLUDE_ALWAYS = [
    "CNTRYID", "CNTRYID_E", "SEQID",
    *[f"SPFWT{i}" for i in range(81)],
    *[f"PVLIT{i}"  for i in range(1, 11)],
    *[f"PVNUM{i}"  for i in range(1, 11)],
    *[f"PVPSL{i}"  for i in range(1, 11)],
    *[f"PVAPS{i}"  for i in range(1, 11)],
    "VEMETHOD", "VEMETHODN", "VEFAYFAC", "VENREPS",
    "VARSTRAT", "VARUNIT",
    "DISP_CIBQ", "DISP_MAIN", "DISP_MAINWRC",
    "RANDOM_CBA_MODULE1", "RANDOM_CBA_MODULE2",
    "RANDOM_CBA_MODULE1_STAGE1", "RANDOM_CBA_MODULE1_STAGE2",
    "RANDOM_CBA_MODULE2_STAGE1", "RANDOM_CBA_MODULE2_STAGE2",
    "RANDOM_PP", "PBROUTE", "PAPER", "CBA_START",
    "CBAMOD1", "CBAMOD2", "CBAMOD2ALT",
    "CBAMOD1STG1", "CBAMOD2STG1",
    "CBAMOD1STG2", "CBAMOD2STG2",
    "MONTHLYINCPR", "YEARLYINCPR",
    "LNG_L1", "LNG_L2", "LNG_HOME", "LNG_BQ", "LNG_CI",
    "CNT_H", "CNT_BRTH", "REG_TL2",
    "CTRYQUAL", "BIRTHRGN", "FIRLGRGN", "SECLGRGN", "HOMLGRGN",
    "D_Q16d1", "D_Q16d2", "D_Q16d3", "D_Q16d4", "D_Q16d5", "D_Q16d6",
]

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

print("=" * 60)
print(f"PIAAC SYNTHETIC DATA GENERATOR  |  Cycle: {CYCLE}  "
      f"|  Mode: {FEATURE_MODE} features")
thr_label = ("none (keep all)" if MISSING_THRESHOLD >= 1.00
             else f"{int(MISSING_THRESHOLD*100)}%")
print(f"Missing threshold: {thr_label}  |  Cycle: {CYCLE}")
print("=" * 60)
print("Loading data …")

with open(DATA_PATH, 'r', encoding='utf-8', errors='replace') as _f:
    _sample = _f.read(4096)
_sep = ';' if _sample.count(';') > _sample.count(',') else ','
print(f"  Detected separator: {repr(_sep)}")

df_raw = pd.read_csv(DATA_PATH, low_memory=False, sep=_sep,
                     encoding='utf-8', encoding_errors='replace')

# Clean column names: strip leading digits, quotes, whitespace
df_raw.columns = [_re.sub('^[0-9]+', '', c).strip() for c in df_raw.columns]
df_raw.columns = [c.strip('"').strip("'").strip() for c in df_raw.columns]

# Deduplicate raw column names immediately (keep first occurrence)
if df_raw.columns.duplicated().any():
    dupes_raw = df_raw.columns[
        df_raw.columns.duplicated(keep=False)].unique().tolist()
    print(f"  WARNING: duplicate raw column(s) — keeping first: {dupes_raw}")
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep='first')]

print(f"  Columns after cleaning: {list(df_raw.columns[:5])} …")
print(f"Raw data shape: {df_raw.shape}")

missing_codes = [9995, 9996, 9997, 9998, 9999, 999999, 99999999]
df_raw = df_raw.replace(['V', ' ', '', 'NA', 'NULL', 'N'], np.nan)
df_raw = df_raw.replace(missing_codes, np.nan)

# ── Derive composite score columns ───────────────────────────────────────────
lit_cols = [c for c in [f"PVLIT{i}" for i in range(1, 11)]
            if c in df_raw.columns]
num_cols = [c for c in [f"PVNUM{i}" for i in range(1, 11)]
            if c in df_raw.columns]
psl_cols = [c for c in [f"PVPSL{i}" for i in range(1, 11)]
            if c in df_raw.columns]

if lit_cols:
    df_raw["LIT_SCORE"] = df_raw[lit_cols].apply(
        pd.to_numeric, errors='coerce').mean(axis=1)
else:
    df_raw["LIT_SCORE"] = np.nan
    print("  WARNING: PVLIT columns not found — LIT_SCORE set to NaN")

if num_cols:
    df_raw["NUM_SCORE"] = df_raw[num_cols].apply(
        pd.to_numeric, errors='coerce').mean(axis=1)
else:
    df_raw["NUM_SCORE"] = np.nan

if psl_cols:
    df_raw["PSL_SCORE"] = df_raw[psl_cols].apply(
        pd.to_numeric, errors='coerce').mean(axis=1)
else:
    df_raw["PSL_SCORE"] = np.nan

# Cycle 2: Adaptive Problem Solving
aps_cols = [c for c in [f"PVAPS{i}" for i in range(1, 11)]
            if c in df_raw.columns]
if aps_cols:
    df_raw["APS_SCORE"] = df_raw[aps_cols].apply(
        pd.to_numeric, errors='coerce').mean(axis=1)
    print(f"  Cycle 2: APS_SCORE derived from {len(aps_cols)} PVAPS columns")
    s = pd.to_numeric(df_raw["APS_SCORE"], errors='coerce')
    df_raw["APSSTATUS"] = pd.cut(
        s, bins=[0, 241, 290, 340, 600], labels=[1, 2, 3, 4]).astype(float)

# ── Cycle 2: derive missing-by-name columns ──────────────────────────────────

# LITSTATUS — derive from LIT_SCORE if absent
if "LITSTATUS" not in df_raw.columns and "LIT_SCORE" in df_raw.columns:
    s = pd.to_numeric(df_raw["LIT_SCORE"], errors='coerce')
    df_raw["LITSTATUS"] = pd.cut(
        s, bins=[0, 176, 226, 276, 326, 600],
        labels=[1, 2, 3, 4, 5]).astype(float)
    print("  Derived: LITSTATUS from LIT_SCORE")

# NUMSTATUS — derive from NUM_SCORE if absent
if "NUMSTATUS" not in df_raw.columns and "NUM_SCORE" in df_raw.columns:
    s = pd.to_numeric(df_raw["NUM_SCORE"], errors='coerce')
    df_raw["NUMSTATUS"] = pd.cut(
        s, bins=[0, 176, 226, 276, 326, 600],
        labels=[1, 2, 3, 4, 5]).astype(float)
    print("  Derived: NUMSTATUS from NUM_SCORE")

# READYTOLEARN — try known Cycle 2 alternative names
if "READYTOLEARN" not in df_raw.columns:
    for _src in ["READYLEARNC2", "READYTOLEARN_C2",
                 "READYTOLEARNC2", "RTL_C2", "READYTOLEARNC2_WLE_CA"]:
        if _src in df_raw.columns:
            df_raw["READYTOLEARN"] = df_raw[_src]
            print(f"  Derived: READYTOLEARN from {_src}")
            break
    else:
        # Last resort: fill with NaN so it is imputed later
        df_raw["READYTOLEARN"] = np.nan
        print("  WARNING: READYTOLEARN not found — will be imputed as median")

# COMPUTEREXPERIENCE — try known Cycle 2 alternative names
if "COMPUTEREXPERIENCE" not in df_raw.columns:
    for _src in ["COMPUTEREXPERIENCEC2", "COMPEXP",
                 "COMPUTEREXP", "COMPEXPC2"]:
        if _src in df_raw.columns:
            df_raw["COMPUTEREXPERIENCE"] = df_raw[_src]
            print(f"  Derived: COMPUTEREXPERIENCE from {_src}")
            break
    else:
        df_raw["COMPUTEREXPERIENCE"] = np.nan
        print("  WARNING: COMPUTEREXPERIENCE not found — will be imputed")

# NFEHRS — Cycle 2 splits into JR/NJR hours; combine if missing
if "NFEHRS" not in df_raw.columns:
    jr  = pd.to_numeric(
        df_raw.get("NFEJRWH",  pd.Series(np.nan, index=df_raw.index)),
        errors='coerce').fillna(0)
    njr = pd.to_numeric(
        df_raw.get("NFENJRWH", pd.Series(np.nan, index=df_raw.index)),
        errors='coerce').fillna(0)
    combined = jr + njr
    df_raw["NFEHRS"] = combined.where(combined > 0, np.nan)
    if combined.sum() > 0:
        print("  Derived: NFEHRS from NFEJRWH + NFENJRWH")
    else:
        print("  WARNING: NFEHRS not found — will be imputed as median")

# Cycle 2 FNFAET12C2
if "FNFAET12C2" not in df_raw.columns:
    fe  = pd.to_numeric(
        df_raw.get("FAET12C2",
                   pd.Series(np.nan, index=df_raw.index)),
        errors='coerce').fillna(0)
    nfe = pd.to_numeric(
        df_raw.get("NFE12C2",
                   pd.Series(np.nan, index=df_raw.index)),
        errors='coerce').fillna(0)
    df_raw["FNFAET12C2"] = ((fe == 1) | (nfe == 1)).astype(int)

# Cycle 2 NFE sub-categories
for col in ["NFE12JRC2", "NFE12NJRC2"]:
    if col not in df_raw.columns:
        df_raw[col] = np.nan

# PLANNINGC2
if "PLANNINGC2" not in df_raw.columns:
    df_raw["PLANNINGC2"] = np.nan

# ── Automatic feature selection ───────────────────────────────────────────────
# 1. Start with every column in the file
# 2. Remove excluded identifiers / weights / routing columns
# 3. Coerce to numeric (non-numeric → NaN)
# 4. Deduplicate column names (keep first occurrence)
# 5. Drop columns that exceed MISSING_THRESHOLD
# 6. Drop all-NaN columns
# 7. If FEATURE_MODE > 0, keep the FEATURE_MODE columns with the lowest
#    missing-value rate (ties broken by original column order)

available = [c for c in df_raw.columns if c not in EXCLUDE_ALWAYS]
print(f"\n  Auto-select: {len(df_raw.columns)} raw columns → "
      f"{len(available)} after excluding identifiers/weights")

df_sel = df_raw[available].apply(pd.to_numeric, errors='coerce')

# Deduplicate
if df_sel.columns.duplicated().any():
    dupes = df_sel.columns[
        df_sel.columns.duplicated(keep=False)].unique().tolist()
    print(f"  WARNING: {len(dupes)} duplicate column name(s) — "
          f"keeping first: {dupes[:8]}")
    df_sel = df_sel.loc[:, ~df_sel.columns.duplicated(keep='first')]

# Drop high-missingness columns
miss_rate = df_sel.isnull().mean()
if MISSING_THRESHOLD < 1.00:
    keep_mask = miss_rate <= MISSING_THRESHOLD
    drop_high = miss_rate[~keep_mask].index.tolist()
    if drop_high:
        print(f"  Dropped {len(drop_high)} columns "
              f"(>{int(MISSING_THRESHOLD*100)}% missing)")
    df_sel = df_sel.loc[:, keep_mask]
else:
    print(f"  MISSING_THRESHOLD=1.00 → keeping all {len(df_sel.columns)} "
          f"columns, imputing all missing values")

# Drop all-NaN columns
all_nan_cols = [c for c in df_sel.columns if bool(df_sel[c].isna().all())]
if all_nan_cols:
    print(f"  Dropped {len(all_nan_cols)} all-NaN columns")
    df_sel = df_sel.drop(columns=all_nan_cols)

# Trim to FEATURE_MODE columns (lowest missing rate → most complete features)
if FEATURE_MODE > 0 and len(df_sel.columns) > FEATURE_MODE:
    miss_sorted = df_sel.isnull().mean().sort_values()
    keep_cols   = miss_sorted.index[:FEATURE_MODE].tolist()
    dropped_n   = len(df_sel.columns) - FEATURE_MODE
    print(f"  Trimmed {dropped_n} columns to reach FEATURE_MODE={FEATURE_MODE} "
          f"(kept columns with lowest missing rate)")
    df_sel = df_sel[keep_cols]
elif FEATURE_MODE == 0:
    print(f"  FEATURE_MODE=0 → keeping all {len(df_sel.columns)} usable columns")

print(f"  Final selection: {len(df_sel.columns)} features")

features = list(df_sel.columns)

imputer  = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(df_sel.values)

if X_imputed.shape[1] != len(features):
    raise ValueError(
        f"Shape mismatch after imputation: array has {X_imputed.shape[1]} "
        f"cols but features list has {len(features)}.")

df_clean   = pd.DataFrame(X_imputed, columns=features)
n_original = len(df_clean)

print(f"\nClean data: {df_clean.shape}")
print(f"Features used ({len(features)}): {features[:10]} …")
if FEATURE_MODE > 0 and len(features) != FEATURE_MODE:
    print(f"  NOTE: running with {len(features)} features "
          f"(FEATURE_MODE={FEATURE_MODE} requested).")

scaler    = StandardScaler()
X         = scaler.fit_transform(df_clean.values)
input_dim = X.shape[1]

if PURE_SYNTHETIC:
    n_to_generate = TARGET_SIZE
else:
    n_to_generate = max(0, TARGET_SIZE - n_original)

if n_to_generate <= 0:
    print(f"\nOriginal data ({n_original}) >= TARGET_SIZE. "
          "No generation needed.")
    raise SystemExit

print(f"\nOriginal rows : {n_original}")
print(f"Target total  : {TARGET_SIZE}")
print(f"Will generate : {n_to_generate} synthetic rows per method")
print(f"Pure synthetic: {PURE_SYNTHETIC}")

# ============================================================
# METHOD 2 — GAN (CTGAN)
# ============================================================

def run_gan(X, n_generate, scaler, features, df):
    print("\n" + "-" * 40)
    print(f"METHOD 2: GAN   (generating {n_generate} samples)")
    print("-" * 40)
    t0 = time.time()
    try:
        from ctgan import CTGAN
        # Cap discrete columns: only include explicitly known categoricals
        # OR columns with <=6 unique values (not <=10, which pulls in too many)
        KNOWN_CATEGORICALS = {
            "GENDER_R", "EDLEVEL3", "READYTOLEARN",
            "LITSTATUS", "NUMSTATUS", "PSLSTATUS", "APSSTATUS",
            "NATIVESPEAKER", "IMGEN", "EDWORK", "EDWORKC2",
            "NEET", "NEETC2", "PAIDWORK12", "LEAVEDU",
            "FE12", "FE12C2", "NFE12", "NFE12C2",
            "FAET12", "FAET12C2", "EARNFLAG", "EARNFLAGC2",
            "VET", "VETC2", "CORESTAGE1_PASS",
            "CORESTAGE2_PASS", "PAPER",
        }
        discrete_cols = [
            c for c in df.columns
            if c in KNOWN_CATEGORICALS or df[c].nunique() <= 6
        ]
        print(f"  Discrete columns ({len(discrete_cols)}): "
              f"{discrete_cols[:8]} …")
        ctgan = CTGAN(
            epochs=100,  # was 200 — halves training time
            batch_size=1000,  # was 500 — larger batches = fewer steps/epoch
            log_frequency=False, verbose=False,
            generator_dim=(128, 128),
            discriminator_dim=(128, 128),
            pac=4,  # reduce PAC from default 10 → 4, faster per step
        )
        ctgan.fit(df, discrete_columns=discrete_cols)
        generated = ctgan.sample(n_generate)
        return {
            'method': 'GAN',
            'data':   generated.values,
            'time':   time.time() - t0,
            'history': [0],
            'final_loss': 0,
        }
    except ImportError:
        print("  CTGAN not installed — skipping.")
        print("  Install with: pip install ctgan")
        return {
            'method': 'GAN',
            'data':   np.zeros((n_generate, len(features))),
            'time':   0.0,
            'history': [0],
            'final_loss': 0,
            '_skipped': True,
        }

# ============================================================
# METHOD 3 — DIFFUSION (DDPM + DDIM)
# ============================================================

def run_diffusion(X, n_generate, scaler, features):
    print("\n" + "-" * 40)
    print(f"METHOD 4: DIFFUSION  (generating {n_generate} samples)")
    print("-" * 40)
    t0        = time.time()
    timesteps = 100
    inp   = layers.Input(shape=(input_dim,))
    x     = layers.Dense(256, activation='relu')(inp)
    x     = layers.BatchNormalization()(x)
    x     = layers.Dense(512, activation='relu')(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.Dense(256, activation='relu')(x)
    out   = layers.Dense(input_dim)(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    beta      = np.linspace(1e-4, 0.02, timesteps).astype(np.float32)
    alpha     = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    X_noisy, X_target = [], []
    for t in range(timesteps):
        ab  = alpha_bar[t]
        eps = np.random.normal(0, 1, X.shape).astype(np.float32)
        X_noisy.append(np.sqrt(ab) * X + np.sqrt(1.0 - ab) * eps)
        X_target.append(eps)
    X_noisy  = np.vstack(X_noisy).astype(np.float32)
    X_target = np.vstack(X_target).astype(np.float32)
    print("  Training …")
    hist = model.fit(X_noisy, X_target, epochs=100, batch_size=256,
                     verbose=0, validation_split=0.1)
    ddim_steps = 20
    step_idx   = np.linspace(0, timesteps-1, ddim_steps, dtype=int)[::-1]
    @tf.function
    def denoise_step(x_t, ab_t, ab_prev):
        eps    = model(x_t, training=False)
        x0_hat = (x_t - tf.sqrt(1.0 - ab_t) * eps) / tf.sqrt(ab_t)
        x0_hat = tf.clip_by_value(x0_hat, -6.0, 6.0)
        return tf.sqrt(ab_prev) * x0_hat + tf.sqrt(1.0 - ab_prev) * eps
    print(f"  Generating {n_generate} samples (DDIM {ddim_steps} steps) …")
    x_gen = tf.constant(
        np.random.normal(0, 1, (n_generate, input_dim)).astype(np.float32))
    for i, t in enumerate(step_idx):
        ab_t    = tf.constant(alpha_bar[t], dtype=tf.float32)
        ab_prev = tf.constant(
            alpha_bar[step_idx[i+1]] if i+1 < len(step_idx) else 1.0,
            dtype=tf.float32)
        x_gen = denoise_step(x_gen, ab_t, ab_prev)
    x_gen = np.clip(x_gen.numpy(), -6.0, 6.0)
    return {
        'method': 'DIFFUSION',
        'data':   scaler.inverse_transform(x_gen),
        'time':   time.time() - t0,
        'history': hist.history['loss'],
        'final_loss': hist.history['loss'][-1],
    }

if _HAS_TORCH:
    # -------------------- Helper functions --------------------
    def _pt_make_loader(X, batch=PT_BATCH_SIZE, shuffle=True):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)

    def _pt_gpu(t):
        return t.to(DEVICE) if torch.cuda.is_available() else t

    # -------------------- MultiHeadDecoder (same as in new_data.py) --------------------
    class MultiHeadDecoder(nn.Module):
        """
        Decoder with separate heads for categorical (one-hot) and numeric features.

        cat_groups : list of (start_idx, end_idx, K)  — one-hot spans in output
        num_dims   : list of column indices for continuous features
        total_dims : total output width (== input_dim)

        The numeric head uses a plain Linear (no sigmoid/tanh) so that
        StandardScaler.inverse_transform can recover arbitrary real values.
        """

        def __init__(self, lat, hid, cat_groups, num_dims, total_dims):
            super().__init__()
            self.cat_groups = cat_groups
            self.num_dims = num_dims
            self.total = total_dims

            # Shared trunk — deeper for better expressivity
            self.trunk = nn.Sequential(
                nn.Linear(lat, hid), nn.LayerNorm(hid), nn.GELU(),
                nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
                nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
                nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),  # extra layer
            )

            # One linear head per categorical span
            self.cat_heads = nn.ModuleList(
                [nn.Linear(hid, K) for _, _, K in cat_groups]
            )

            # Numeric head — NO sigmoid, outputs unconstrained reals
            if num_dims:
                nh = max(hid, 256)
                self.num_trunk = nn.Sequential(
                    nn.Linear(lat, nh), nn.LayerNorm(nh), nn.GELU(),
                    nn.Linear(nh, nh), nn.LayerNorm(nh), nn.GELU(),
                    nn.Linear(nh, nh), nn.GELU(),
                )
                # Linear final layer — no activation
                self.num_head = nn.Linear(nh, len(num_dims))
            else:
                self.num_trunk = self.num_head = None

        def forward(self, z):
            h = self.trunk(z)
            out = torch.zeros(z.size(0), self.total, device=z.device)

            # Categorical logits → raw (used in training with cross-entropy or MSE)
            for head, (s, e, K) in zip(self.cat_heads, self.cat_groups):
                out[:, s:e] = head(h)

            # Numeric → unconstrained linear output
            if self.num_head is not None:
                idx = torch.tensor(self.num_dims, device=z.device, dtype=torch.long)
                out[:, idx] = self.num_head(self.num_trunk(z))
            return out

        @torch.no_grad()
        def sample(self, z, temperature=PT_CAT_TEMP):
            h = self.trunk(z)
            out = torch.zeros(z.size(0), self.total, device=z.device)

            for head, (s, e, K) in zip(self.cat_heads, self.cat_groups):
                logits = head(h) / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                oh = F.one_hot(
                    torch.multinomial(probs, 1).squeeze(-1), K
                ).float()
                out[:, s:e] = oh

            if self.num_head is not None:
                idx = torch.tensor(self.num_dims, device=z.device, dtype=torch.long)
                # Linear output — no activation, no clip (clipping happens in postprocess)
                out[:, idx] = self.num_head(self.num_trunk(z))
            return out

    # -------------------- CVAE Encoder --------------------
    class _Res(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d))
        def forward(self, x):
            return x + self.ff(self.ln(x))

    class CVAEEnc(nn.Module):
        def __init__(self, in_d, hid, lat):
            super().__init__()
            mid = min(hid*2, 1024)
            self.proj = nn.Sequential(nn.Linear(in_d, mid), nn.LayerNorm(mid), nn.GELU(),
                                       nn.Linear(mid, hid), nn.LayerNorm(hid), nn.GELU())
            self.res = nn.Sequential(*[_Res(hid) for _ in range(6)])
            self.norm = nn.LayerNorm(hid)
            self.mu = nn.Linear(hid, lat)
            self.lv = nn.Linear(hid, lat)
        def forward(self, x):
            h = self.norm(self.res(self.proj(x)))
            return self.mu(h), self.lv(h).clamp(-4, 4)

    # -------------------- CVAE Model --------------------
    class CVAE(nn.Module):
        def __init__(self, in_d, cat_groups, num_dims, hid=PT_HIDDEN_DIM, lat=PT_LATENT_DIM):
            super().__init__()
            self.enc = CVAEEnc(in_d, hid, lat)
            self.dec = MultiHeadDecoder(lat, hid, cat_groups, num_dims, in_d)
            self.lat = lat
        def _rp(self, mu, lv):
            return mu + torch.exp(0.5 * lv.clamp(-10, 10)) * torch.randn_like(mu)
        def forward(self, x):
            mu, lv = self.enc(x)
            return self.dec(self._rp(mu, lv)), mu, lv
        @torch.no_grad()
        def sample(self, n):
            self.eval()
            return self.dec.sample(torch.randn(n, self.lat, device=next(self.parameters()).device)).cpu().numpy()
        @torch.no_grad()
        def encode_mu(self, x):
            return self.enc(x)[0]

    # -------------------- VAE training loop --------------------
    def _train_vae_core(model, X, epochs, lr=PT_LR, tag="VAE"):
        """
        Trains with β-TC-VAE loss for DisentangledCVAE:
          total loss = recon + β_kl * KL + β_tc * TC
        TC penalty encourages the three latent subspaces to be independent,
        which is exactly what we need for HiDe-Tab's conditional diffusion.
        For plain CVAE (3-tuple output) falls back to standard β-VAE.
        """
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.05)
        loader = _pt_make_loader(X)
        best_loss, best_state = float("inf"), None

        for ep in range(1, epochs + 1):
            model.train()
            total_recon = 0.0

            for (xb,) in loader:
                xb = xb.to(DEVICE)
                outputs = model(xb)
                recon = outputs[0]
                recon_loss = F.mse_loss(recon, xb)

                if isinstance(outputs[1], tuple):
                    kl = 0.0
                    for (mu_i, lv_i) in (outputs[1], outputs[2], outputs[3]):
                        kl += -0.5 * torch.mean(1 + lv_i - mu_i.pow(2) - lv_i.exp())

                    mu_s, lv_s = outputs[1]
                    mu_v, lv_v = outputs[2]
                    mu_c, lv_c = outputs[3]
                    # Detach means for TC only — pushes heads apart without
                    # corrupting the shared trunk's reconstruction gradient
                    mu_s_d = mu_s.detach()
                    mu_v_d = mu_v.detach()
                    mu_c_d = mu_c.detach()
                    min_sv = min(mu_s_d.size(1), mu_v_d.size(1))
                    min_vc = min(mu_v_d.size(1), mu_c_d.size(1))
                    tc_sv = (mu_s_d[:, :min_sv] * mu_v_d[:, :min_sv]).pow(2).mean()
                    tc_vc = (mu_v_d[:, :min_vc] * mu_c_d[:, :min_vc]).pow(2).mean()
                    tc = tc_sv + tc_vc

                    loss = recon_loss + 0.05 * kl + 0.05 * tc
                else:
                    mu, lv = outputs[1], outputs[2]
                    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                    loss = recon_loss + 0.1 * kl

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_recon += recon_loss.item()

            sch.step()
            avg = total_recon / len(loader)
            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ep % 50 == 0 or ep == epochs:
                print(f"      [{tag}] ep {ep:4d}/{epochs} recon={avg:.5f}")

        model.load_state_dict(
            {k: v.to(DEVICE) for k, v in best_state.items()})
        return model

    # ==================== HiDe‑Tab ====================
    # (Simplified version: treat all features as numeric, cat_groups empty)
    class DisentangledEncoder(nn.Module):
        def __init__(self, in_d, hid, lat_s, lat_v, lat_c):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_d, hid), nn.LayerNorm(hid), nn.GELU(),
            )
            self.res_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hid),
                    nn.Linear(hid, hid * 2), nn.GELU(), nn.Dropout(0.05),
                    nn.Linear(hid * 2, hid),
                ) for _ in range(4)
            ])
            self.head_proj_s = nn.Sequential(nn.Linear(hid, hid), nn.GELU())
            self.head_proj_v = nn.Sequential(nn.Linear(hid, hid), nn.GELU())
            self.head_proj_c = nn.Sequential(nn.Linear(hid, hid), nn.GELU())
            self.mu_s = nn.Linear(hid, lat_s); self.lv_s = nn.Linear(hid, lat_s)
            self.mu_v = nn.Linear(hid, lat_v); self.lv_v = nn.Linear(hid, lat_v)
            self.mu_c = nn.Linear(hid, lat_c); self.lv_c = nn.Linear(hid, lat_c)

        def forward(self, x):
            h = self.proj(x)
            for blk in self.res_blocks:
                h = h + blk(h)
            # No .detach() — let reconstruction loss train the shared trunk.
            # Cross-head disentanglement is enforced by the TC penalty in the
            # loss function, which operates on the means AFTER forward pass.
            hs = self.head_proj_s(h)
            hv = self.head_proj_v(h)
            hc = self.head_proj_c(h)
            return (
                (self.mu_s(hs), self.lv_s(hs).clamp(-4, 4)),
                (self.mu_v(hv), self.lv_v(hv).clamp(-4, 4)),
                (self.mu_c(hc), self.lv_c(hc).clamp(-4, 4)),
            )

        def reparameterize(self, mu, lv):
            return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)


    class DisentangledDecoder(nn.Module):
        def __init__(self, lat_total, hid, total_dims):
            super().__init__()
            self.decoder = MultiHeadDecoder(
                lat_total, hid, [], list(range(total_dims)), total_dims)

        def forward(self, z):
            return self.decoder(z)

        def sample(self, z, temp=PT_CAT_TEMP):
            return self.decoder.sample(z, temp)


    class DisentangledCVAE(nn.Module):
        def __init__(self, in_d, hid=PT_HIDDEN_DIM,
                     lat_s=PT_LAT_S, lat_v=PT_LAT_V, lat_c=PT_LAT_C):
            super().__init__()
            self.enc = DisentangledEncoder(in_d, hid, lat_s, lat_v, lat_c)
            # Decoder sees [z_s | z_v | z_c] — fixed order
            self.dec = DisentangledDecoder(lat_s + lat_v + lat_c, hid, in_d)
            self.lat_s_dim = lat_s
            self.lat_v_dim = lat_v
            self.lat_c_dim = lat_c

        def forward(self, x):
            (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c) = self.enc(x)
            z_s = self.enc.reparameterize(mu_s, lv_s)
            z_v = self.enc.reparameterize(mu_v, lv_v)
            z_c = self.enc.reparameterize(mu_c, lv_c)
            z = torch.cat([z_s, z_v, z_c], dim=1)
            return self.dec(z), (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c), (z_s, z_v, z_c)

        def encode(self, x):
            (mu_s, lv_s), (mu_v, lv_v), (mu_c, lv_c) = self.enc(x)
            # Return concatenated means + individual tuples for KL
            return (torch.cat([mu_s, mu_v, mu_c], dim=1),
                    (mu_s, mu_v, mu_c),
                    (lv_s, lv_v, lv_c))


    class DiffNetMLP(nn.Module):
        """
        Noise predictor with FiLM-style conditioning.
        Instead of additive cond_proj, we use scale+shift (FiLM) so the
        conditioning can modulate both the magnitude and bias of each layer.
        This is much more effective than simple addition for conditional diffusion.
        """

        def __init__(self, in_dim, hid=PT_HIDDEN_DIM, n_blocks=4, cond_dim=0):
            super().__init__()
            self.cond_dim = cond_dim

            # Sinusoidal-style time embedding
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hid), nn.SiLU(),
                nn.Linear(hid, hid), nn.SiLU(),
            )

            # FiLM conditioning: produces (scale, shift) per block
            # Each block gets its own scale+shift projection
            if cond_dim > 0:
                self.film_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(cond_dim, hid), nn.SiLU(),
                        nn.Linear(hid, hid * 2),  # → [scale | shift]
                    ) for _ in range(n_blocks)
                ])
            else:
                self.film_layers = None

            self.input_proj = nn.Linear(in_dim, hid)

            # Residual blocks with pre-norm
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hid),
                    nn.Linear(hid, hid * 2), nn.GELU(), nn.Dropout(0.05),
                    nn.Linear(hid * 2, hid),
                ) for _ in range(n_blocks)
            ])
            self.out_norm = nn.LayerNorm(hid)
            self.out = nn.Linear(hid, in_dim)

        def forward(self, x, t, cond=None):
            t_f = t.float().unsqueeze(-1) / PT_T_STEPS
            t_emb = self.time_mlp(t_f)
            h = self.input_proj(x) + t_emb

            for i, blk in enumerate(self.blocks):
                residual = blk(h)
                if self.film_layers is not None and cond is not None:
                    # FiLM: scale and shift the residual using conditioning
                    film = self.film_layers[i](cond)
                    scale, shift = film.chunk(2, dim=-1)
                    # scale centred at 1 for stable training
                    residual = residual * (1 + scale) + shift
                h = h + residual

            return self.out(self.out_norm(h))


    class DDPMScheduler:
        def __init__(self, T=PT_T_STEPS, s=0.008, device='cpu'):
            self.T      = T
            self.device = device
            steps = torch.arange(T + 1, dtype=torch.float32)
            f     = torch.cos(((steps / T) + s) / (1 + s) * np.pi * 0.5) ** 2
            f     = f / f[0]
            betas = (1.0 - f[1:] / f[:-1]).clamp(1e-5, 0.999)
            self.betas     = betas.to(device)
            self.alpha_bar = f[1:].to(device)

        def q_sample(self, x0, t):
            ab  = self.alpha_bar[t].view(-1, 1)
            eps = torch.randn_like(x0)
            return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps

        @torch.no_grad()
        def ddim_sample(self, model, shape, device, ddim_steps=50, cond=None):
            step_idx = torch.linspace(self.T - 1, 0, ddim_steps,
                                      dtype=torch.long, device=device)
            x = torch.randn(shape, device=device)

            for i in range(len(step_idx)):
                t_cur   = step_idx[i]
                is_last = (i + 1 == len(step_idx))  # Python bool — never ambiguous

                tb  = t_cur.expand(shape[0])
                eps = model(x, tb, cond) if cond is not None else model(x, tb)

                ab = self.alpha_bar[t_cur]
                # KEY FIX: was `self.alpha_bar[t_prev] if t_prev >= 0 else ...`
                # t_prev was torch.tensor(-1), so `>= 0` returned tensor(False)
                # which is truthy (non-zero tensor), so alpha_bar[-1] = alpha_bar[T-1]
                # injecting noise at the final step. Use Python bool instead.
                ab_prev = (torch.tensor(1.0, device=device) if is_last
                           else self.alpha_bar[step_idx[i + 1]])

                x0_pred = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-5, 5)
                x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps

            return x

        @torch.no_grad()
        def p_sample_loop(self, model, shape, device, cond=None):
            x = torch.randn(shape, device=device)
            for i in reversed(range(self.T)):
                tb      = torch.full((shape[0],), i, device=device, dtype=torch.long)
                eps     = model(x, tb) if cond is None else model(x, tb, cond)
                ab      = self.alpha_bar[i]
                ab_prev = (self.alpha_bar[i - 1] if i > 0
                           else torch.tensor(1.0, device=device))
                x0h  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-5, 5)
                if i == 0:
                    # Algorithm line: z_v^(0) ← x0_pred (no noise at final step)
                    x = x0h
                else:
                    beta = self.betas[i]
                    mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0h + \
                           (ab.sqrt() * (1 - ab_prev) / (1 - ab)) * x
                    var  = (beta * (1 - ab_prev) / (1 - ab)).clamp(1e-20)
                    x    = mean + var.sqrt() * torch.randn_like(x)
            return x


    def run_hidetab(X, n_generate, scaler, features):
        print("\n" + "-" * 40)
        print(f"METHOD: HiDe-Tab  (generating {n_generate} samples)")
        print("-" * 40)
        t0 = time.time()
        D  = X.shape[1]
        X_t = torch.tensor(X, dtype=torch.float32)

        # Stage 1: Disentangled VAE
        vae = DisentangledCVAE(in_d=D).to(DEVICE)
        _train_vae_core(vae, X_t, epochs=PT_EPOCHS_VAE, lr=PT_LR, tag="HiDe-VAE")
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        # Encode
        Z_sc_list, Z_v_list = [], []
        with torch.no_grad():
            for (xb,) in _pt_make_loader(X_t, batch=512, shuffle=False):
                xb = xb.to(DEVICE)
                _, (mu_s, mu_v, mu_c), _ = vae.encode(xb)
                Z_sc_list.append(torch.cat([mu_s, mu_c], dim=1).cpu())
                Z_v_list.append(mu_v.cpu())

        Z_sc = torch.cat(Z_sc_list, dim=0)
        Z_v  = torch.cat(Z_v_list,  dim=0)

        Z_sc_mean = Z_sc.mean(0, keepdim=True)
        Z_sc_std  = Z_sc.std(0,  keepdim=True).clamp(1e-3)
        Z_v_mean  = Z_v.mean(0,  keepdim=True)
        Z_v_std   = Z_v.std(0,   keepdim=True).clamp(1e-3)
        Zn_sc = (Z_sc - Z_sc_mean) / Z_sc_std
        Zn_v  = (Z_v  - Z_v_mean)  / Z_v_std

        sc_dim = Zn_sc.shape[1]
        v_dim  = Zn_v.shape[1]
        lat_s  = PT_LAT_S

        # Stage A: Transformer denoiser on z_sc (macro, unconditional, T steps)
        # Matches Algorithm 1: "Macro Diffusion (Transformer denoiser)"
        class TransformerDenoiser(nn.Module):
            """Single-token Transformer denoiser for z_sc (unconditional)."""
            def __init__(self, lat_d, hid=PT_HIDDEN_DIM, n_heads=4, n_layers=3):
                super().__init__()
                self.time_mlp = nn.Sequential(
                    nn.Linear(1, hid), nn.SiLU(),
                    nn.Linear(hid, hid), nn.SiLU(),
                )
                self.input_proj = nn.Linear(lat_d, hid)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=hid, nhead=n_heads, dim_feedforward=hid * 2,
                    dropout=0.1, batch_first=True, norm_first=True)
                self.transformer = nn.TransformerEncoder(enc_layer,
                                                         num_layers=n_layers)
                self.out_proj = nn.Linear(hid, lat_d)

            def forward(self, z, t, cond=None):
                t_emb = self.time_mlp(t.float().unsqueeze(-1) / PT_T_STEPS)
                h = self.input_proj(z).unsqueeze(1) + t_emb.unsqueeze(1)
                h = self.transformer(h)
                return self.out_proj(h.squeeze(1))

        model_A = TransformerDenoiser(lat_d=sc_dim).to(DEVICE)
        sch_A   = DDPMScheduler(T=PT_T_STEPS, device=DEVICE)
        opt_A   = torch.optim.AdamW(model_A.parameters(), lr=PT_LR * 2.0,
                                     weight_decay=1e-4)
        sched_A = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_A, PT_EPOCHS_DIFF, eta_min=PT_LR * 0.05)

        # EMA for Stage A
        ema_A_shadow = {k: v.cpu().clone().float()
                        for k, v in model_A.state_dict().items()}

        loader_A = _pt_make_loader(Zn_sc, batch=PT_BATCH_SIZE, shuffle=True)
        print(f"      Stage A: Transformer denoiser on z_sc (dim={sc_dim})")
        for ep in range(1, PT_EPOCHS_DIFF + 1):
            model_A.train()
            total = 0.0
            for (zb,) in loader_A:
                zb    = zb.to(DEVICE)
                t_idx = torch.randint(0, PT_T_STEPS, (zb.size(0),), device=DEVICE)
                zt, noise = sch_A.q_sample(zb, t_idx)
                loss = F.mse_loss(model_A(zt, t_idx), noise)
                opt_A.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
                opt_A.step()
                # EMA update
                for k, v in model_A.state_dict().items():
                    ema_A_shadow[k] = 0.9999 * ema_A_shadow[k] + 0.0001 * v.cpu().float()
                total += loss.item()
            sched_A.step()
            if ep % 50 == 0 or ep == PT_EPOCHS_DIFF:
                print(f"      Stage A ep {ep:4d}/{PT_EPOCHS_DIFF} "
                      f"loss={total / len(loader_A):.5f}")
        # Apply EMA weights for sampling
        model_A.load_state_dict(
            {k: v.to(DEVICE) for k, v in ema_A_shadow.items()})
        model_A.eval()

        # Stage B: 1D-CNN denoiser conditioned on z_sc via FiLM
        class DiffNet1DCNN(nn.Module):
            """
            1D-convolutional denoiser for z_v conditioned on z_sc.
            z_v is treated as a 1-D sequence of length lat_v with 1 channel.
            FiLM scale+shift from z_sc is applied after every conv block.
            """
            def __init__(self, lat_v, cond_dim, hid=256, n_blocks=4):
                super().__init__()
                self.time_mlp = nn.Sequential(
                    nn.Linear(1, hid), nn.SiLU(),
                    nn.Linear(hid, hid), nn.SiLU(),
                )
                # FiLM: maps cond + time to (scale, shift) per channel
                self.film_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(cond_dim, hid), nn.SiLU(),
                        nn.Linear(hid, hid * 2),      # → [scale | shift]
                    ) for _ in range(n_blocks)
                ])
                self.input_conv = nn.Conv1d(1, hid, kernel_size=3, padding=1)
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(hid, hid, kernel_size=3, padding=1),
                        nn.GroupNorm(8, hid),
                        nn.SiLU(),
                    ) for _ in range(n_blocks)
                ])
                self.out_conv = nn.Sequential(
                    nn.Conv1d(hid, hid // 2, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(hid // 2, 1, kernel_size=3, padding=1),
                )
                self.final_proj = nn.Linear(lat_v, lat_v)

            def forward(self, z_v, t, cond):
                # time embedding (B, hid)
                t_emb = self.time_mlp(t.float().unsqueeze(-1) / PT_T_STEPS)
                # (B, 1, lat_v) → (B, hid, lat_v)
                h = self.input_conv(z_v.unsqueeze(1))
                for i, blk in enumerate(self.blocks):
                    h_res = blk(h)
                    # FiLM conditioning from z_sc (+ absorb time via addition)
                    film = self.film_proj[i](cond + t_emb)
                    scale, shift = film.chunk(2, dim=-1)     # each (B, hid)
                    h_res = h_res * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
                    h = h + h_res                            # residual
                out = self.out_conv(h).squeeze(1)            # (B, lat_v)
                return self.final_proj(out)

        model_B = DiffNet1DCNN(lat_v=v_dim, cond_dim=sc_dim,
                                hid=256, n_blocks=4).to(DEVICE)
        sch_B   = DDPMScheduler(T=PT_T_STEPS // 2, device=DEVICE)
        opt_B   = torch.optim.AdamW(model_B.parameters(), lr=PT_LR,
                                     weight_decay=1e-5)
        sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_B, PT_EPOCHS_DIFF, eta_min=PT_LR * 0.05)

        # EMA for Stage B
        ema_B_shadow = {k: v.cpu().clone().float()
                        for k, v in model_B.state_dict().items()}

        loader_B = DataLoader(TensorDataset(Zn_v, Zn_sc),
                              batch_size=PT_BATCH_SIZE, shuffle=True,
                              num_workers=0)
        print(f"      Stage B: 1D-CNN denoiser on z_v | z_sc (dim={v_dim}, "
              f"T={sch_B.T})")
        for ep in range(1, PT_EPOCHS_DIFF + 1):
            model_B.train()
            total = 0.0
            for z_v_b, z_sc_b in loader_B:
                z_v_b  = z_v_b.to(DEVICE)
                z_sc_b = z_sc_b.to(DEVICE)
                t_idx  = torch.randint(0, sch_B.T, (z_v_b.size(0),),
                                       device=DEVICE)
                zt, noise = sch_B.q_sample(z_v_b, t_idx)
                pred  = model_B(zt, t_idx, z_sc_b)
                loss  = F.mse_loss(pred, noise)
                opt_B.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
                opt_B.step()
                # EMA update
                for k, v in model_B.state_dict().items():
                    ema_B_shadow[k] = 0.9999 * ema_B_shadow[k] + 0.0001 * v.cpu().float()
                total += loss.item()
            sched_B.step()
            if ep % 50 == 0 or ep == PT_EPOCHS_DIFF:
                print(f"      Stage B ep {ep:4d}/{PT_EPOCHS_DIFF} "
                      f"loss={total / len(loader_B):.5f}")
        # Apply EMA weights for sampling
        model_B.load_state_dict(
            {k: v.to(DEVICE) for k, v in ema_B_shadow.items()})
        model_B.eval()

        # ── Sampling (matches Algorithm 1) ────────────────────────────────────
        # Stage A: DDIM reverse process using EMA weights of model_A
        # Stage B: DDPM reverse process using EMA weights of model_B
        # Conditioning: model_B receives normalized z_sc (Zn_sc_cond)
        #   because model_B was trained on normalized Zn_sc as conditioning signal
        # Decoder: z_full = [z_s; z_v; z_c]  (fixed VAE decoder order)
        Z_sc_mean_d = Z_sc_mean.to(DEVICE); Z_sc_std_d = Z_sc_std.to(DEVICE)
        Z_v_mean_d  = Z_v_mean.to(DEVICE);  Z_v_std_d  = Z_v_std.to(DEVICE)

        all_samples = []
        gen_batch   = 512

        for start in range(0, n_generate, gen_batch):
            cur = min(gen_batch, n_generate - start)

            # Stage A: DDIM 50-step reverse process for z_sc (T=PT_T_STEPS)
            # Result is normalized; denormalize for decoding
            z_sc_norm = sch_A.ddim_sample(model_A, (cur, sc_dim), DEVICE,
                                           ddim_steps=50, cond=None)
            z_sc = z_sc_norm * Z_sc_std_d + Z_sc_mean_d

            # Stage B: full DDPM reverse for z_v | z_sc (T=PT_T_STEPS//2)
            # Pass normalized z_sc as conditioning (matches training distribution)
            z_v_norm = sch_B.p_sample_loop(model_B, (cur, v_dim), DEVICE,
                                            cond=z_sc_norm)
            z_v = z_v_norm * Z_v_std_d + Z_v_mean_d

            # Decode: split z_sc back into z_s and z_c
            z_s    = z_sc[:, :lat_s]
            z_c    = z_sc[:, lat_s:]
            z_full = torch.cat([z_s, z_v, z_c], dim=1)

            with torch.no_grad():
                x_syn = vae.dec.sample(z_full, PT_CAT_TEMP)
            all_samples.append(x_syn.cpu().numpy())

        gen = np.vstack(all_samples).astype(np.float32)
        return {
            'method':     'HiDe_Tab',
            'data':       scaler.inverse_transform(gen),
            'time':       time.time() - t0,
            'history':    [0],
            'final_loss': 0,
        }

    # ==================== TabDiff (simplified) ====================
    class TabDiffVAE(nn.Module):
        def __init__(self, in_d, hid=PT_HIDDEN_DIM, lat=PT_LATENT_DIM):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(in_d, hid), nn.LayerNorm(hid), nn.GELU(),
                nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
                nn.Linear(hid, lat))
            self.dec = nn.Sequential(
                nn.Linear(lat, hid), nn.LayerNorm(hid), nn.GELU(),
                nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
                nn.Linear(hid, in_d))
        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    def run_tabdiff(X, n_generate, scaler, features):
        print("\n" + "-" * 40)
        print(f"METHOD: TabDiff  (generating {n_generate} samples)")
        print("-" * 40)
        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.tensor(X, dtype=torch.float32)
        model = TabDiffVAE(in_d=X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=PT_LR)
        loader = _pt_make_loader(X_t)
        for ep in range(1, PT_EPOCHS_VAE+1):
            model.train()
            total_loss = 0.0
            for (xb,) in loader:
                xb = _pt_gpu(xb)
                recon, _ = model(xb)
                loss = F.mse_loss(recon, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if ep % 50 == 0:
                print(f"      TabDiff ep {ep:4d}/{PT_EPOCHS_VAE} loss={total_loss/len(loader):.5f}")
        model.eval()
        with torch.no_grad():
            z = torch.randn(n_generate, PT_LATENT_DIM, device=device)
            gen = model.dec(z).cpu().numpy()
        return {
            'method': 'TabDiff',
            'data': scaler.inverse_transform(gen),
            'time': time.time() - t0,
            'history': [0],
            'final_loss': 0,
        }

    # ==================== TabSyn ====================
    class TabSyn(nn.Module):
        def __init__(self, in_d, lat=PT_TABSYN_LATENT, hid=PT_HIDDEN_DIM):
            super().__init__()
            self.enc = CVAEEnc(in_d, hid, lat)
            self.dec = MultiHeadDecoder(lat, hid, [], list(range(in_d)), in_d)
        def forward(self, x):
            mu, lv = self.enc(x)
            z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
            return self.dec(z), mu, lv
        @torch.no_grad()
        def encode(self, x):
            return self.enc(x)[0]
        @torch.no_grad()
        def decode(self, z):
            return self.dec.sample(z, PT_CAT_TEMP)

    def run_tabsyn(X, n_generate, scaler, features):
        print("\n" + "-" * 40)
        print(f"METHOD: TabSyn  (generating {n_generate} samples)")
        print("-" * 40)
        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.tensor(X, dtype=torch.float32)
        # Stage 1: VAE
        vae = TabSyn(in_d=X.shape[1]).to(device)
        _train_vae_core(vae, X_t, epochs=PT_EPOCHS_VAE, lr=PT_LR, tag="TabSyn-VAE")
        vae.eval()
        # Encode to latent
        Z_list = []
        with torch.no_grad():
            for batch in _pt_make_loader(X_t, batch=256):
                xb = _pt_gpu(batch[0])
                z = vae.encode(xb)
                Z_list.append(z.cpu())
        Z = torch.cat(Z_list, dim=0)
        Z_mean, Z_std = Z.mean(0, keepdim=True), Z.std(0, keepdim=True).clamp(1e-3)
        Zn = (Z - Z_mean) / Z_std
        # Stage 2: Diffusion on latent
        diff_model = DiffNetMLP(in_dim=Zn.shape[1]).to(device)
        sch = DDPMScheduler(device=device)
        opt = torch.optim.AdamW(diff_model.parameters(), lr=PT_LR)
        for ep in range(1, PT_EPOCHS_DIFF+1):
            diff_model.train()
            total = 0.0
            for (zb,) in _pt_make_loader(Zn):
                zb = _pt_gpu(zb)
                t = torch.randint(0, PT_T_STEPS, (zb.size(0),), device=device)
                zt, noise = sch.q_sample(zb, t)
                loss = F.mse_loss(diff_model(zt, t), noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            if ep % 50 == 0:
                print(f"      TabSyn diff ep {ep:4d}/{PT_EPOCHS_DIFF} loss={total/len(Zn):.5f}")
        diff_model.eval()
        # Sample
        vae.to(device); diff_model.to(device); Z_mean_d = Z_mean.to(device); Z_std_d = Z_std.to(device)
        all_samples = []
        with torch.no_grad():
            for start in range(0, n_generate, 256):
                cur = min(256, n_generate - start)
                z = sch.p_sample_loop(diff_model, (cur, Zn.shape[1]), device) * Z_std_d + Z_mean_d
                x_syn = vae.decode(z).cpu().numpy()
                all_samples.append(x_syn)
        gen = np.vstack(all_samples).astype(np.float32)
        return {
            'method': 'TabSyn',
            'data': scaler.inverse_transform(gen),
            'time': time.time() - t0,
            'history': [0],
            'final_loss': 0,
        }

else:
    # Dummy functions when PyTorch is missing
    def run_hidetab(*args, **kwargs):
        print("PyTorch not installed – HiDe-Tab skipped.")
        return {'method': 'HiDe_Tab', 'data': np.zeros((0,0)), 'time':0, 'history':[0], 'final_loss':0, '_skipped':True}
    def run_tabdiff(*args, **kwargs):
        print("PyTorch not installed – TabDiff skipped.")
        return {'method': 'TabDiff', 'data': np.zeros((0,0)), 'time':0, 'history':[0], 'final_loss':0, '_skipped':True}
    def run_tabsyn(*args, **kwargs):
        print("PyTorch not installed – TabSyn skipped.")
        return {'method': 'TabSyn', 'data': np.zeros((0,0)), 'time':0, 'history':[0], 'final_loss':0, '_skipped':True}

# ============================================================
# POSTPROCESSING
# ============================================================

CLIP_RULES = {
    "AGE_R":              (16,  65,   True),
    "GENDER_R":           (1,   2,    True),
    "EDLEVEL3":           (1,   6,    True),
    "EDCAT7":             (1,   7,    True),
    "EDCAT7_TC1":         (1,   7,    True),
    "EDCAT8":             (1,   8,    True),
    "EDCAT8_TC1":         (1,   8,    True),
    "EDCAT6_TC1":         (1,   6,    True),
    "READYTOLEARN":       (1,   5,    True),
    "ICTHOME":            (0,   5,    False),
    "ICTHOMEC2":          (0,   5,    False),
    "ICTWORK":            (0,   5,    False),
    "ICTWORKC2":          (0,   5,    False),
    "LEARNATWORK":        (0,   5,    False),
    "LEARNATWORKC2":      (0,   5,    False),
    "INFLUENCE":          (0,   5,    False),
    "INFLUENCEC2_T1":     (0,   5,    False),
    "NUMHOME":            (0,   5,    False),
    "NUMHOMEC2":          (0,   5,    False),
    "NUMWORK":            (0,   5,    False),
    "NUMWORKC2":          (0,   5,    False),
    "PLANNING":           (0,   5,    False),
    "PLANNINGC2":         (0,   5,    False),
    "READHOME":           (0,   5,    False),
    "READHOMEC2_T1":      (0,   5,    False),
    "READWORK":           (0,   5,    False),
    "READWORKC2_T1":      (0,   5,    False),
    "TASKDISC":           (0,   5,    False),
    "TASKDISCC2_T1":      (0,   5,    False),
    "WRITHOME":           (0,   5,    False),
    "WRITHOMEC2":         (0,   5,    False),
    "WRITWORK":           (0,   5,    False),
    "WRITWORKC2":         (0,   5,    False),
    "NATIVESPEAKER":      (1,   2,    True),
    "IMGEN":              (1,   3,    True),
    "IMGENC2":            (1,   3,    True),
    "LITSTATUS":          (1,   5,    True),
    "NUMSTATUS":          (1,   5,    True),
    "PSLSTATUS":          (1,   4,    True),
    "APSSTATUS":          (1,   4,    True),
    "COMPUTEREXPERIENCE": (1,   5,    True),
    "PARED":              (1,   6,    True),
    "PAREDC2":            (1,   6,    True),
    "ISCOSKIL4":          (1,   4,    True),
    "NEET":               (0,   1,    True),
    "NEETC2":             (0,   1,    True),
    "PAIDWORK12":         (0,   1,    True),
    "EARNFLAG":           (0,   1,    True),
    "EARNFLAGC2":         (0,   1,    True),
    "FE12":               (0,   1,    True),
    "FE12C2":             (0,   1,    True),
    "NFE12":              (0,   1,    True),
    "NFE12C2":            (0,   1,    True),
    "FAET12":             (0,   1,    True),
    "FAET12C2":           (0,   1,    True),
    "LEAVEDU":            (1,   4,    True),
    "VET":                (0,   1,    True),
    "VETC2":              (0,   1,    True),
    "CORESTAGE1_PASS":    (0,   1,    True),
    "CORESTAGE2_PASS":    (0,   1,    True),
    "LIT_SCORE":          (0,   500,  False),
    "NUM_SCORE":          (0,   500,  False),
    "PSL_SCORE":          (0,   500,  False),
    "APS_SCORE":          (0,   500,  False),
    "EARNMTH":            (0,   None, False),
    "EARNMTHC2":          (0,   None, False),
    "EARNMTHALL":         (0,   None, False),
    "EARNMTHALLC2":       (0,   None, False),
    "EARNHR":             (0,   None, False),
    "EARNHRC2":           (0,   None, False),
    "YRSQUAL":            (0,   25,   False),
    "YRSQUALC2":          (0,   25,   False),
    "NFEHRS":             (0,   None, False),
    "ISCED_HF":           (0,   8,    True),
    "IMYRS_C":            (0,   5,    True),
    "PARTNER":            (0,   1,    True),
    "CHILDNR":            (0,   None, False),
    "AGRE":               (0,   None, False),
    "CONS":               (0,   None, False),
    "EMOS":               (0,   None, False),
    "EXTR":               (0,   None, False),
    "OPEM":               (0,   None, False),
}

def postprocess(df_synth):
    df_pp = df_synth.copy()
    for col, (lo, hi, do_round) in CLIP_RULES.items():
        if col not in df_pp.columns:
            continue
        if hi is not None:
            df_pp[col] = df_pp[col].clip(lo, hi)
        else:
            df_pp[col] = df_pp[col].clip(lower=lo)
        if do_round:
            df_pp[col] = df_pp[col].round().astype(int)
    return df_pp

# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(real_df, synth_df, method_name):
    """
    Evaluate synthetic data quality using three complementary dimensions.

    Statistical fidelity (per-column):
        KS statistic   — marginal distribution similarity (lower = better)
        Wasserstein     — earth-mover distance (lower = better)
        Jensen-Shannon  — symmetric KL divergence on binned histograms
        Cohen's d       — standardised mean difference (lower = better)
    Correlation fidelity:
        Mean |Δcorr|    — average absolute difference between real and
                          synthetic Pearson correlation matrices (lower = better)
    ML utility (Train-on-Synthetic / Test-on-Real paradigm):
        Classification  — binary target: literacy score > median
                          (above/below median split on real data)
                          Metric: AUC-ROC (Random Forest, 100 trees)
        Regression      — continuous target: literacy score (same column)
                          Metric: R² on held-out 20% of real data
        Three conditions tested: real-trained, synth-trained, mixed-trained
    """
    print(f"  Evaluating {method_name} …")
    results = {
        'method': method_name,
        'statistical': {}, 'ml_classification': {},
        'ml_regression': {}, 'correlation': {},
    }
    stats_list = []
    for col in real_df.columns:
        r = real_df[col].dropna()
        s = synth_df[col].dropna()
        if len(r) < 10 or len(s) < 10:
            continue
        ks, _  = ks_2samp(r, s)
        wass   = wasserstein_distance(r, s)
        bins   = np.histogram_bin_edges(
            np.concatenate([r, s]), bins=30)
        hr, _  = np.histogram(r, bins=bins, density=True)
        hs, _  = np.histogram(s, bins=bins, density=True)
        hr += 1e-10; hs += 1e-10
        jsd    = jensenshannon(hr, hs)
        pooled = np.sqrt((r.std()**2 + s.std()**2) / 2)
        cd     = (r.mean() - s.mean()) / (pooled + 1e-8)
        stats_list.append({
            'feature': col, 'ks': ks, 'wasserstein': wass,
            'jsd': jsd, 'cohen_d': cd,
            'real_mean': r.mean(), 'synth_mean': s.mean(),
            'real_std': r.std(), 'synth_std': s.std(),
        })
    results['statistical'] = stats_list
    results['statistical_avg'] = {
        'ks_avg':          np.mean([s['ks']          for s in stats_list]),
        'wasserstein_avg': np.mean([s['wasserstein']  for s in stats_list]),
        'jsd_avg':         np.mean([s['jsd']          for s in stats_list]),
        'cohen_d_avg':     np.mean([abs(s['cohen_d']) for s in stats_list]),
    }
    corr_real  = real_df.corr()
    corr_synth = synth_df.corr()
    corr_diff  = (corr_real - corr_synth).abs()
    results['correlation'] = {
        'mean_abs_diff': corr_diff.mean().mean(),
        'max_abs_diff':  corr_diff.max().max(),
    }
    # Use LIT_SCORE if present, otherwise APS_SCORE
    score_col = None
    for candidate in ["LIT_SCORE", "APS_SCORE", "NUM_SCORE"]:
        if candidate in real_df.columns:
            score_col = candidate
            break
    if score_col is None:
        results['ml_classification'] = {
            k: {'accuracy': 0, 'f1': 0, 'auc': 0}
            for k in ['real', 'synthetic', 'mixed']}
        results['ml_regression'] = {
            k: {'rmse': 0, 'r2': 0}
            for k in ['real', 'synthetic', 'mixed']}
        return results
    ml_feats   = [c for c in real_df.columns if c != score_col]
    median_val = real_df[score_col].median()
    y_real     = (real_df[score_col]  > median_val).astype(int)
    y_synth    = (synth_df[score_col] > median_val).astype(int)
    Xr_clf = real_df[ml_feats]
    Xs_clf = synth_df[ml_feats]
    # 80/20 stratified split — stratify keeps class balance in test set
    # test_size=0.2 and random_state=RANDOM_SEED ensure reproducibility
    Xtr, Xte, ytr, yte = train_test_split(
        Xr_clf, y_real, test_size=0.2,
        random_state=RANDOM_SEED, stratify=y_real)
    def clf_scores(X_train, y_train):
        # Random Forest: 100 trees, fixed seed, all CPU cores
        clf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        return {'accuracy': accuracy_score(yte, pred),
                'f1': f1_score(yte, pred),
                'auc': roc_auc_score(yte, prob)}
    results['ml_classification'] = {
        'real':      clf_scores(Xtr, ytr),
        'synthetic': clf_scores(Xs_clf, y_synth),
        'mixed':     clf_scores(
            pd.concat([Xtr, Xs_clf]),
            pd.concat([ytr, y_synth])),
    }
    yr = real_df[score_col]
    ys = synth_df[score_col]
    # Regression: same 80/20 split, fixed seed
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        real_df[ml_feats], yr, test_size=0.2, random_state=RANDOM_SEED)
    def reg_scores(X_train, y_train):
        reg = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        reg.fit(X_train, y_train)
        pred = reg.predict(Xte2)
        return {'rmse': np.sqrt(mean_squared_error(yte2, pred)),
                'r2': r2_score(yte2, pred)}
    results['ml_regression'] = {
        'real':      reg_scores(Xtr2, ytr2),
        'synthetic': reg_scores(synth_df[ml_feats], ys),
        'mixed':     reg_scores(
            pd.concat([Xtr2, synth_df[ml_feats]]),
            pd.concat([ytr2, ys])),
    }
    # ── Privacy metric: Distance to Closest Record (DCR) ───────────────────
    # DCR measures the minimum Euclidean distance from each synthetic record
    # to any real record (in standardised feature space).
    # A low DCR may indicate memorisation; DCR ≈ real-to-real distances
    # (5th-percentile baseline) suggests adequate privacy preservation.
    try:
        from sklearn.preprocessing import StandardScaler as _SS
        _sc = _SS()
        Xr_norm = _sc.fit_transform(real_df.values.astype(float))
        Xs_norm = _sc.transform(synth_df.values.astype(float))
        # Compute pairwise distances in batches to avoid OOM
        batch = 500
        dcr_vals = []
        for i in range(0, len(Xs_norm), batch):
            chunk = Xs_norm[i:i+batch]
            dists = np.sqrt(((chunk[:, None, :] - Xr_norm[None, :, :]) ** 2).sum(axis=2))
            dcr_vals.extend(dists.min(axis=1).tolist())
        # Real-to-real baseline (sample 1000 pairs)
        rr_idx = np.random.choice(len(Xr_norm),
                                   min(1000, len(Xr_norm)), replace=False)
        rr_dists = []
        for idx in rr_idx[:100]:
            d = np.sqrt(((Xr_norm[idx] - Xr_norm) ** 2).sum(axis=1))
            d_no_self = d[d > 0]
            if len(d_no_self):
                rr_dists.append(d_no_self.min())
        results['privacy'] = {
            'dcr_mean':          float(np.mean(dcr_vals)),
            'dcr_p5':            float(np.percentile(dcr_vals, 5)),
            'dcr_median':        float(np.median(dcr_vals)),
            'rr_baseline_p5':    float(np.percentile(rr_dists, 5)) if rr_dists else 0.0,
            'privacy_ratio':     float(np.percentile(dcr_vals, 5) /
                                       (np.percentile(rr_dists, 5) + 1e-8)),
            # ratio > 1 → synth data is farther from real than real records
            # are from each other (good privacy signal)
        }
        print(f"    DCR p5={results['privacy']['dcr_p5']:.4f}  "
              f"RR-baseline p5={results['privacy']['rr_baseline_p5']:.4f}  "
              f"ratio={results['privacy']['privacy_ratio']:.3f}")
    except Exception as _pe:
        results['privacy'] = {'error': str(_pe)}
    return results

# ============================================================
# REPEATED EVALUATION — mean ± std over N_EVAL_RUNS seeds
# ============================================================

def evaluate_method_repeated(real_df, synth_df, method_name, n_runs=N_EVAL_RUNS):
    """
    Run evaluate_method N_EVAL_RUNS times with different random seeds
    and return mean ± std for AUC and R² (key ML utility metrics).
    Statistical metrics (KS, JSD, Cohen's d) are deterministic given
    the same data so they are only computed once (seed=RANDOM_SEED).
    """
    # Deterministic metrics — compute once
    base = evaluate_method(real_df, synth_df, method_name)
    auc_runs, r2_runs = [], []
    score_col = None
    for candidate in ["LIT_SCORE", "APS_SCORE", "NUM_SCORE"]:
        if candidate in real_df.columns:
            score_col = candidate
            break
    if score_col is None:
        base['auc_std'] = 0.0
        base['r2_std']  = 0.0
        return base
    ml_feats   = [c for c in real_df.columns if c != score_col]
    median_val = real_df[score_col].median()
    y_real     = (real_df[score_col] > median_val).astype(int)
    yr         = real_df[score_col]
    y_synth    = (synth_df[score_col] > median_val).astype(int)
    for run_seed in range(RANDOM_SEED, RANDOM_SEED + n_runs):
        Xtr, Xte, ytr, yte = train_test_split(
            real_df[ml_feats], y_real, test_size=0.2,
            random_state=run_seed, stratify=y_real)
        clf = RandomForestClassifier(
            n_estimators=100, random_state=run_seed, n_jobs=-1)
        clf.fit(synth_df[ml_feats], y_synth)
        prob = clf.predict_proba(Xte)[:, 1]
        auc_runs.append(roc_auc_score(yte, prob))
        Xtr2, Xte2, ytr2, yte2 = train_test_split(
            real_df[ml_feats], yr, test_size=0.2, random_state=run_seed)
        reg = RandomForestRegressor(
            n_estimators=100, random_state=run_seed, n_jobs=-1)
        reg.fit(synth_df[ml_feats], synth_df[score_col])
        pred2 = reg.predict(Xte2)
        r2_runs.append(r2_score(yte2, pred2))
    base['auc_mean'] = float(np.mean(auc_runs))
    base['auc_std']  = float(np.std(auc_runs))
    base['r2_mean']  = float(np.mean(r2_runs))
    base['r2_std']   = float(np.std(r2_runs))
    print(f"    {method_name}: AUC={base['auc_mean']:.4f}±{base['auc_std']:.4f}  "
          f"R²={base['r2_mean']:.4f}±{base['r2_std']:.4f}  "
          f"(over {n_runs} seeds)")
    return base

# ============================================================
# RUN ALL METHODS (including the three new ones)
# ============================================================
print("\n" + "=" * 60)
print(f"RUNNING ALL GENERATIVE METHODS  |  {n_to_generate} samples each")
print("=" * 60)

method_funcs = [run_gan, run_diffusion]
if _HAS_TORCH:
    method_funcs += [run_hidetab, run_tabdiff, run_tabsyn]

all_results        = []
all_synthetic_data = {}
training_histories = {}

for i, fn in enumerate(method_funcs):
    try:
        print(f"\n{'='*60}\nMETHOD {i+1} / {len(method_funcs)}\n{'='*60}")
        if fn.__name__ == 'run_gan':
            result = fn(X, n_to_generate, scaler, features, df_clean)
        else:
            result = fn(X, n_to_generate, scaler, features)
        if result.get('_skipped'):
            print(f"  ⚠ {result['method']} was skipped.")
            continue
        synth_df = pd.DataFrame(result['data'], columns=features)
        synth_df = postprocess(synth_df)
        tag = result['method']
        synth_df.to_csv(
            f"outputs/data/synthetic_{tag}_mode{FEATURE_MODE}.csv",
            index=False)
        all_synthetic_data[tag] = synth_df
        training_histories[tag] = {
            'loss': result['history'],
            'final_loss': result['final_loss'],
            'time': result['time'],
        }
        print(f"\n  Evaluating …")
        eval_res = evaluate_method_repeated(df_clean, synth_df, tag, N_EVAL_RUNS)
        eval_res['training_time'] = result['time']
        eval_res['final_loss']    = result['final_loss']
        eval_res['n_generated']   = len(synth_df)
        all_results.append(eval_res)
        print(f"  ✓ {tag}: {len(synth_df)} samples  "
              f"|  {result['time']:.1f}s  "
              f"|  loss {result['final_loss']:.4f}")
    except Exception as exc:
        import traceback
        print(f"  ✗ ERROR in {fn.__name__}: {exc}")
        traceback.print_exc()

# ============================================================
# COMBINED DATASETS
# ============================================================

print("\n" + "=" * 60 + "\nCREATING COMBINED DATASETS\n" + "=" * 60)
for tag, synth_df in all_synthetic_data.items():
    if PURE_SYNTHETIC:
        final_df = synth_df.copy()
        final_df['_source'] = 'synthetic'
    else:
        real_tagged  = df_clean.copy()
        real_tagged['_source']  = 'real'
        synth_tagged = synth_df.copy()
        synth_tagged['_source'] = 'synthetic'
        final_df = pd.concat(
            [real_tagged, synth_tagged], ignore_index=True)
    out_path = (f"outputs/data/final_combined_{tag}"
                f"_mode{FEATURE_MODE}.csv")
    final_df.to_csv(out_path, index=False)
    print(f"  {tag}: {final_df.shape}  → {out_path}")

# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60 + "\nCOMPARISON SUMMARY\n" + "=" * 60)
rows = []
for res in all_results:
    rows.append({
        'Method':          res['method'],
        'Features':        len(features),
        'Generated':       res['n_generated'],
        'Time (s)':        f"{res['training_time']:.1f}",
        'Final Loss':      f"{res.get('final_loss', 0):.4f}",
        'Avg KS':          f"{res['statistical_avg']['ks_avg']:.4f}",
        'Avg Wasserstein': f"{res['statistical_avg']['wasserstein_avg']:.3f}",
        'Avg JSD':         f"{res['statistical_avg']['jsd_avg']:.4f}",
        'Avg |Cohen d|':   f"{res['statistical_avg']['cohen_d_avg']:.4f}",
        'Corr Diff':       f"{res['correlation']['mean_abs_diff']:.4f}",
        'Clf AUC (Synth)': f"{res['ml_classification']['synthetic']['auc']:.4f}",
        'AUC mean±std':    f"{res.get('auc_mean', res['ml_classification']['synthetic']['auc']):.4f}"
                           f"±{res.get('auc_std', 0):.4f}",
        'Reg R² (Synth)':  f"{res['ml_regression']['synthetic']['r2']:.4f}",
        'R² mean±std':     f"{res.get('r2_mean', res['ml_regression']['synthetic']['r2']):.4f}"
                           f"±{res.get('r2_std', 0):.4f}",
        'Reg R² (Mixed)':  f"{res['ml_regression']['mixed']['r2']:.4f}",
        'DCR p5':          f"{res.get('privacy', {}).get('dcr_p5', float('nan')):.4f}",
        'Privacy ratio':   f"{res.get('privacy', {}).get('privacy_ratio', float('nan')):.3f}",
    })
comp_df = pd.DataFrame(rows)
comp_df.to_csv(
    f"outputs/comparison/method_comparison_mode{FEATURE_MODE}.csv",
    index=False)
print(comp_df.to_string(index=False))

# ============================================================
# VISUALISATIONS
# ============================================================

print("\nGenerating plots …")

plt.figure(figsize=(12, 5))
for tag, hist in training_histories.items():
    if len(hist['loss']) > 1:
        plt.plot(hist['loss'],
                 label=f"{tag} (final={hist['final_loss']:.4f})")
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title(f'Training Loss  |  Mode {FEATURE_MODE} features')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(
    f"outputs/comparison/training_loss_mode{FEATURE_MODE}.png",
    dpi=200, bbox_inches='tight')
plt.close()

metrics     = ['ks_avg', 'jsd_avg', 'cohen_d_avg']
metric_lbls = ['Avg KS', 'Avg JSD', 'Avg |Cohen d|']
fig, axes   = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Statistical Metrics  |  Mode {FEATURE_MODE} features')
for ax, met, lbl in zip(axes, metrics, metric_lbls):
    vals    = [r['statistical_avg'][met] for r in all_results]
    methods = [r['method'] for r in all_results]
    bars    = ax.bar(methods, vals)
    ax.set_title(lbl); ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=35)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + max(vals) * 0.02,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(
    f"outputs/comparison/statistical_metrics_mode{FEATURE_MODE}.png",
    dpi=200, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'ML Utility  |  Mode {FEATURE_MODE} features')
methods = [r['method'] for r in all_results]
x = np.arange(len(methods)); w = 0.25

def bar3(ax, rv, sv, mv, title, ylabel):
    ax.bar(x - w, rv, w, label='Real-trained',  alpha=0.8)
    ax.bar(x,     sv, w, label='Synth-trained', alpha=0.8)
    ax.bar(x + w, mv, w, label='Mixed-trained', alpha=0.8)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35)
    ax.legend(); ax.grid(alpha=0.3)

bar3(axes[0, 0],
     [r['ml_classification']['real']['auc']      for r in all_results],
     [r['ml_classification']['synthetic']['auc'] for r in all_results],
     [r['ml_classification']['mixed']['auc']     for r in all_results],
     'Classification AUC', 'AUC')
bar3(axes[0, 1],
     [r['ml_regression']['real']['r2']      for r in all_results],
     [r['ml_regression']['synthetic']['r2'] for r in all_results],
     [r['ml_regression']['mixed']['r2']     for r in all_results],
     'Regression R²', 'R²')
corr_vals = [r['correlation']['mean_abs_diff'] for r in all_results]
bars = axes[1, 0].bar(methods, corr_vals)
axes[1, 0].set_title('Correlation Diff (lower=better)')
axes[1, 0].set_ylabel('Mean |Δ corr|')
axes[1, 0].tick_params(axis='x', rotation=35)
for b, v in zip(bars, corr_vals):
    axes[1, 0].text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(corr_vals) * 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
times = [r['training_time'] for r in all_results]
bars  = axes[1, 1].bar(methods, times)
axes[1, 1].set_title('Training Time')
axes[1, 1].set_ylabel('Seconds')
axes[1, 1].tick_params(axis='x', rotation=35)
for b, v in zip(bars, times):
    axes[1, 1].text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(times) * 0.02,
                    f'{v:.0f}s', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(
    f"outputs/comparison/ml_performance_mode{FEATURE_MODE}.png",
    dpi=200, bbox_inches='tight')
plt.close()

key_plot_features = [
    f for f in ["AGE_R", "GENDER_R", "EDLEVEL3", "LIT_SCORE",
                "NUM_SCORE", "APS_SCORE", "EARNMTHC2", "ICTHOMEC2"]
    if f in df_clean.columns
][:6]

for feat in key_plot_features:
    n_methods = len(all_synthetic_data)
    ncols = min(3, n_methods + 1)
    nrows = (n_methods + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows))
    fig.suptitle(
        f'Distribution: {feat}  |  Mode {FEATURE_MODE}', fontsize=14)
    axs = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    axs[0].hist(df_clean[feat], bins=30, density=True,
                alpha=0.8, color='steelblue', label='Real')
    axs[0].set_title('Real Data')
    axs[0].set_xlabel(feat); axs[0].legend()
    for j, (tag, synth_df) in enumerate(all_synthetic_data.items()):
        ax = axs[j + 1]
        ax.hist(df_clean[feat], bins=30, density=True,
                alpha=0.5, color='steelblue', label='Real')
        ax.hist(synth_df[feat], bins=30, density=True,
                alpha=0.5, color='darkorange', label='Synthetic')
        ax.set_title(tag); ax.set_xlabel(feat); ax.legend()
    for k in range(n_methods + 1, len(axs)):
        axs[k].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        f"outputs/comparison/dist_{feat}_mode{FEATURE_MODE}.png",
        dpi=150, bbox_inches='tight')
    plt.close()

hm_features = features[:20]
n_methods   = len(all_synthetic_data)
ncols = min(3, n_methods + 1)
nrows = (n_methods + 1 + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(7 * ncols, 6 * nrows))
fig.suptitle(
    f'Correlation (first 20 features)  |  Mode {FEATURE_MODE}',
    fontsize=14)
axs = axes.flatten() if hasattr(axes, 'flatten') else [axes]
sns.heatmap(df_clean[hm_features].corr(), cmap='coolwarm',
            center=0, ax=axs[0], cbar=True,
            xticklabels=True, yticklabels=True)
axs[0].set_title('Real')
axs[0].tick_params(axis='x', rotation=45, labelsize=6)
axs[0].tick_params(axis='y', labelsize=6)
for j, (tag, synth_df) in enumerate(all_synthetic_data.items()):
    ax = axs[j + 1]
    sns.heatmap(synth_df[hm_features].corr(), cmap='coolwarm',
                center=0, ax=ax, cbar=True,
                xticklabels=True, yticklabels=True)
    ax.set_title(tag)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
for k in range(n_methods + 1, len(axs)):
    axs[k].set_visible(False)
plt.tight_layout()
plt.savefig(
    f"outputs/comparison/correlation_heatmaps_mode{FEATURE_MODE}.png",
    dpi=150, bbox_inches='tight')
plt.close()

pca      = PCA(n_components=2)
real_pca = pca.fit_transform(df_clean.values)
ncols = min(3, n_methods + 1)
nrows = (n_methods + 1 + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6 * ncols, 5 * nrows))
fig.suptitle(f'PCA  |  Mode {FEATURE_MODE}', fontsize=14)
axs = axes.flatten() if hasattr(axes, 'flatten') else [axes]
axs[0].scatter(real_pca[:, 0], real_pca[:, 1],
               s=5, alpha=0.5, c='steelblue', label='Real')
axs[0].set_title('Real'); axs[0].legend()
for j, (tag, synth_df) in enumerate(all_synthetic_data.items()):
    ax    = axs[j + 1]
    s_pca = pca.transform(synth_df.values)
    ax.scatter(real_pca[:, 0], real_pca[:, 1],
               s=5, alpha=0.3, c='steelblue', label='Real')
    ax.scatter(s_pca[:, 0], s_pca[:, 1],
               s=5, alpha=0.3, c='darkorange', label='Synthetic')
    ax.set_title(tag); ax.legend(markerscale=3)
for k in range(n_methods + 1, len(axs)):
    axs[k].set_visible(False)
plt.tight_layout()
plt.savefig(f"outputs/comparison/pca_mode{FEATURE_MODE}.png",
            dpi=150, bbox_inches='tight')
plt.close()

print("  Running t-SNE …")
sample_n   = min(1000, n_original)
rng_tsne   = np.random.default_rng(RANDOM_SEED)
real_idx   = rng_tsne.choice(n_original, sample_n, replace=False)
tsne_data  = [df_clean.values[real_idx]]
tsne_meths = ['Real'] * sample_n
for tag, synth_df in all_synthetic_data.items():
    s_idx = min(sample_n, len(synth_df))
    tsne_data.append(synth_df.values[:s_idx])
    tsne_meths += [tag] * s_idx
tsne_combined = np.vstack(tsne_data)
tsne_res = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30,
                n_iter=1000, method='barnes_hut').fit_transform(
                    tsne_combined)
plt.figure(figsize=(12, 8))
unique_tags = list(dict.fromkeys(tsne_meths))
colours     = plt.cm.tab10(np.linspace(0, 1, len(unique_tags)))
cmap        = dict(zip(unique_tags, colours))
for tag in unique_tags:
    mask = np.array(tsne_meths) == tag
    plt.scatter(tsne_res[mask, 0], tsne_res[mask, 1],
                label=tag, s=8, alpha=0.6, c=[cmap[tag]])
plt.title(f't-SNE  |  Mode {FEATURE_MODE} features')
plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"outputs/comparison/tsne_mode{FEATURE_MODE}.png",
            dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# RANKING
# ============================================================

print("\n" + "=" * 60 + "\nMETHOD RANKING\n" + "=" * 60)
ranking = pd.DataFrame({
    'Method':  [r['method'] for r in all_results],
    'Stat':    [1 / (r['statistical_avg']['ks_avg'] + 0.001)
                for r in all_results],
    'Corr':    [1 / (r['correlation']['mean_abs_diff'] + 0.001)
                for r in all_results],
    'ML':      [r.get('auc_mean', r['ml_classification']['synthetic']['auc'])
                + max(0, r.get('r2_mean', r['ml_regression']['synthetic']['r2']))
                for r in all_results],
    'Privacy': [r.get('privacy', {}).get('privacy_ratio', 0.5)
                for r in all_results],
    'Speed':   [1 / (r['training_time'] + 1) for r in all_results],
})
for col in ['Stat', 'Corr', 'ML', 'Privacy', 'Speed']:
    rng = ranking[col].max() - ranking[col].min()
    ranking[col] = ((ranking[col] - ranking[col].min())
                    / (rng + 1e-8))
ranking['Overall'] = ranking[['Stat', 'Corr', 'ML', 'Privacy', 'Speed']].mean(axis=1)
ranking = ranking.sort_values('Overall', ascending=False)
print(ranking[['Method', 'Overall']].to_string(index=False))
best_method = ranking.iloc[0]['Method']
print(f"\n→ Best overall: {best_method}")

if best_method in all_synthetic_data:
    best_df = all_synthetic_data[best_method].copy()
    best_df['_source']            = 'synthetic'
    best_df['_generation_method'] = best_method
    best_df['_feature_mode']      = FEATURE_MODE
    best_df['_generation_date']   = pd.Timestamp.now().isoformat()
    best_path = (f"outputs/data/BEST_synthetic_piaac"
                 f"_{best_method}_mode{FEATURE_MODE}.csv")
    best_df.to_csv(best_path, index=False)
    print(f"  Saved: {best_path}")

json_results = {
    'config': {
        'feature_mode':      FEATURE_MODE,
        'cycle':             CYCLE,
        'target_size':       TARGET_SIZE,
        'pure_synthetic':    PURE_SYNTHETIC,
        'missing_threshold': MISSING_THRESHOLD,
        'random_seed':       RANDOM_SEED,
        'n_eval_runs':       N_EVAL_RUNS,
        'n_original':        n_original,
        'n_generated':       n_to_generate,
        'features_used':     features,
        'n_features':        len(features),
        'tf_version':        tf.__version__,
        'sklearn_version':   __import__('sklearn').__version__,
        'pandas_version':    pd.__version__,
        'numpy_version':     np.__version__,
    },
    'method_comparison': comp_df.to_dict('records'),
    'ranking':           ranking.to_dict('records'),
}
json_path = f"outputs/comparison/results_mode{FEATURE_MODE}.json"
with open(json_path, 'w') as f:
    json.dump(json_results, f, indent=2, default=str)

print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)
print(f"  Cycle             : {CYCLE}")
print(f"  Feature mode      : {FEATURE_MODE}")
print(f"  Missing threshold : {thr_label}")
print(f"  Features used     : {len(features)}")
print(f"  Original rows     : {n_original}")
print(f"  Generated/method  : {n_to_generate}")
print(f"  Pure synthetic    : {PURE_SYNTHETIC}")
print("\nOutput files:")
print("  outputs/data/        — synthetic & combined CSVs")
print("  outputs/comparison/  — plots, table, JSON")
print("=" * 60)