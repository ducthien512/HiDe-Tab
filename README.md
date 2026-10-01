# HiDe-Tab: Hierarchical Disentangled Diffusion for Tabular Data Generation

Implementation of **HiDe-Tab** evaluated on two Czech sociological datasets and four public sociological benchmarks.

---

## Files

| File | Dataset |
|------|---------|
| `PIAAC dataset_HiDe_Tab.py` | PIAAC Czech (Cycle 1 & 2) |
| `Kult2012 dataset_HiDe_Tab.py` | Kult2012 (private) |
| `Extra dataset_HiDe-Tab.py` | Adult, German Credit, ESS R10, GSS 2022 |

---

## Datasets

### PIAAC Czech — publicly available

| Cycle | Rows | Columns |
|-------|------|---------|
| Cycle 1 (2011–2012) | 6,102 | 1,328 |
| Cycle 2 (2022–2023) | 5,057 | 2,483 |

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

### Four public sociological datasets

| Dataset | Rows | Features | Target | Positive rate | Download |
|---------|------|----------|--------|---------------|----------|
| Adult (Census Income) | 32,561 | 13 | income >50K | ~24% | [UCI](https://archive.ics.uci.edu/dataset/2/adult) |
| German Credit | 1,000 | 20 | bad credit risk | ~30% | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| ESS Round 10 (2020) | 37,611 | ~30 | life satisfaction | ~49% | [ESS portal](https://ess-search.nsd.no/en/study/172ac431-2a06-41df-9dab-c1fd8f3877e7) |
| GSS 2022 | ~4,000 | ~50 | general happiness | ~30% | [NORC](https://gss.norc.org/Get-The-Data) |

All four are freely available. ESS and GSS require free registration to download.

**Place files as:**
```
data/
├── adult/adult.data
├── german/german.data
├── ESS10/ESS10.csv
└── GSS2022/GSS2022.dta
```

---

## Requirements

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn torch
pip install tensorflow    # PIAAC only
pip install pyreadstat    # Kult2012 only (.sav reading)
```

A CUDA GPU is recommended. All scripts fall back to CPU automatically.

---

## Usage

### PIAAC

Edit the top of `PIAAC dataset_HiDe_Tab.py`:

```python
DATA_PATH    = "PIAAC PUF dataset\\Cycle 1\\prgczep1.csv"
CYCLE        = 1        # 1 or 2
FEATURE_MODE = 1000     # 500 or 1000
TARGET_SIZE  = 5000     # 5000 / 10000 / 20000
```

```bash
python PIAAC dataset_HiDe_Tab.py
```

### Kult2012

Edit the top of `Kult2012 dataset_HiDe_Tab.py`:

```python
DATA_PATH = r"path\to\Kult2012_....sav"
N_SYNTH   = 5000		# 5000 / 10000 / 20000
```

```bash
python Kult2012 dataset_HiDe_Tab.py
```

### Four public datasets

Edit `DATA_PATHS` and `DATASETS_TO_RUN` at the top of `Extra dataset_HiDe-Tab.py`:

```python
METHODS  = ["CTGAN", "Diffusion", "HiDe_Tab", "TabDiff", "TabSyn"]
N_SYNTH  = 20000
```

```bash
python Extra dataset_HiDe-Tab.py
```

---

## Methods

| Method | Description |
|--------|-------------|
| **HiDe-Tab** | Proposed: disentangled VAE (TC penalty) + two-stage hierarchical diffusion (Transformer macro → 1D-CNN micro via FiLM) |
| CTGAN | Conditional tabular GAN |
| Diffusion | Flat latent diffusion baseline |
| TabDiff | Mixed-type diffusion |
| TabSyn | Latent score-based diffusion |

---

## License

Code released for academic research. PIAAC PUF files are subject to OECD terms of use. Kult2012 data are private and not included.
