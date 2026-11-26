# Stress biaxiality selects how graphene fails — data and code

Data, fine-tuned models and scripts for:

**M. Attia, F. Djefaflia**, *Stress biaxiality, not loading direction, selects how graphene fails: an actively learned foundation potential from phonon instabilities to finite-temperature strength* (submitted).

## Contents
| Folder | Content |
|---|---|
| `data/` | DFT training/test sets (`train.xyz`, `test.xyz`, extxyz; 333/64 structures, PBE, Quantum ESPRESSO), raw DFT single points (`dft_configs/`), held-out DFT frozen-phonon states (`frozen_phonon/`) |
| `models/` | Fine-tuned MACE-OMAT-0 models: round 0 (`ft_omat_s1/s2`), round 1 (`r1_s*`), round 2 (`r2_s1`, `r2_s2`, used in the paper) |
| `scripts/` | Data generation, fine-tuning (`finetune.sh`), stress-controlled paths, phonons, crossover map, MD, DFT checks, figures |
| `results/` | Benchmark of foundation models, phonon scans and λ*(θ), MD tensile runs, DFT validation |
| `figures/` | Figures (PDF/PNG) and the plotted data as CSV (`figures/data/`, OriginPro-ready) |

## Usage
```python
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths="models/ft_omat_r2_s2.model", default_dtype="float64")
```
Energies follow the MACE-OMAT-0 reference; to convert to the Quantum ESPRESSO energy scale add the per-atom shift in `results/validation/eref.json`.

DFT settings: PBE, SSSP pseudopotentials, 50/400 Ry, Marzari–Vanderbilt 0.01 Ry, vacuum 12 Å, k density equivalent to 24×24 for the primitive cell. 2D stress = σ·L_z (N/m), positive in tension.

Scripts use absolute paths (`/home/user/graphene`); adjust them to your setup before running.

## License
Code: MIT. Data and models: CC BY 4.0.
