# HiDe-Tab: Hierarchical Disentangled Diffusion for Tabular Data Generation

Implementation of **HiDe-Tab** evaluated on two Czech sociological datasets.

---

## Files

| File | Dataset |
|------|---------|
| `PIAAC_dataset_HiDe_Tab.py` | PIAAC Czech (Cycle 1 & 2) |
| `Kult2012_dataset_HiDe_Tab.py` | Kult2012 (private) |

---

## Datasets

### PIAAC Czech — publicly available

Download the Czech Public Use Files (PUF) from the OECD:
**<https://www.oecd.org/skills/piaac/data/>**

Place files as:
```
PIAAC PUF dataset/
├── Cycle 1/prgczep1.csv
└── Cycle 2/prgczep2.csv
```

### Kult2012 — private

A Czech social survey (3,679 respondents, 275 variables) covering cultural participation across three regions. The `.sav` file is not publicly available. Researchers wishing to access it should contact the data custodian directly.

---

## Requirements

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn torch
pip install tensorflow    # PIAAC only
pip install pyreadstat    # Kult2012 only (.sav reading)
```

A CUDA GPU is recommended. Both scripts fall back to CPU automatically.

---

## Usage

### PIAAC

Edit the top of `PIAAC_dataset_HiDe_Tab.py`:

```python
DATA_PATH    = "PIAAC PUF dataset\\Cycle 1\\prgczep1.csv"
CYCLE        = 1        # 1 / 2
FEATURE_MODE = 1000     # 500 / 1000
TARGET_SIZE  = 5000     # 5000 / 10000 / 20000
```

```bash
python PIAAC dataset_HiDe_Tab.py
```

### Kult2012

Edit the top of `Kult2012_dataset_HiDe_Tab.py`:

```python
DATA_PATH = r"path\to\Kult2012_....sav"
N_SYNTH   = 5000  # 5000 / 10000 / 20000

METHODS = [
    "HiDe_Tab",          # full model
    #"CTGAN",            # baselines
    #"Diffusion",
    #"TabDiff",
    #"TabSyn",
    #"HiDeTab_noTC",     # ablations
    #"HiDeTab_1Stage",
    #"HiDeTab_3Stage",
    #"HiDeTab_noFiLM",
]
```

```bash
python Kult2012 dataset_HiDe_Tab.py
```

## License

Code released for academic research. PIAAC PUF files are subject to OECD terms of use. Kult2012 data are private and not included.
