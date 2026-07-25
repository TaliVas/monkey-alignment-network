# Deep Learning Framework for Kinematic Event Detection and Stimulation Decoding in Primate Reaching Behavior

Code and subset data for the paper:

> A. Markus, N. Sinha, Y. Prut, and J. Goldberger,
> *"Deep learning framework for kinematic event detection and stimulation decoding in primate reaching behavior,"* 2026.
> Preprint: bioRxiv (Manuscript ID: BIORXIV/2026/738537).
> A preliminary version of the inter-animal alignment method appeared in:
> A. Markus, N. Sinha, Y. Prut, and J. Goldberger, *"Automatic inter-animal alignment of recorded kinematic trajectories,"* IEEE ICASSP, 2026.

This repository provides code to (1) detect kinematic events (movement onset and corrective turning points) from single-trial 3D reaching trajectories using a bidirectional LSTM (BiLSTM), (2) align recordings across animals with an orthogonal Procrustes (Kabsch) transform, and (3) decode control vs. high-frequency-stimulation (HFS) trials from single-trial kinematics.

---

## Repository structure

```
monkey-alignment-network/
├── src/                         # Notebooks and scripts
│   ├── matlab_to_dataframe.ipynb
│   ├── alignment.ipynb
│   ├── compute_features.ipynb
│   ├── model_train.ipynb
│   ├── onset_model_train_eval.py
│   ├── turning_model_train_eval.py
│   ├── turn_10_times_no_align.py
│   └── hfs_decoding_train_eval.py
├── features/                    # Subset feature files (200 trials each)
├── requirements.txt
├── CITATION.cff
├── LICENSE                      # GPL-3.0
└── README.md
```

## Installation

```bash
git clone https://github.com/TaliVas/monkey-alignment-network.git
cd monkey-alignment-network
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```
Tested with Python 3.10. Main dependencies: PyTorch, NumPy, pandas, SciPy, scikit-learn, joblib, Jupyter.

## Data availability

This repository contains **subset feature files with 200 trials each** (the full dataset is 10,000–13,000+ trials per animal). The subsets reproduce the full pipeline at reduced size. The **complete dataset is available upon reasonable request.**

Subset files in `features/`:
- `Features_nana_subset`, `Features_nana_aligned_subset`
- `Features_thina_subset`, `Features_thina_aligned_subset`

For the decoding task, class-balanced control/HFS subsets are provided in `features/` (see the decoding script).

## Pipeline

1. **MATLAB → DataFrame** (`src/matlab_to_dataframe.ipynb`): convert raw `.mat` to CSV.
2. **Inter-animal alignment** (`src/alignment.ipynb`): orthogonal Procrustes/Kabsch alignment of the two animals' coordinate frames.
3. **Feature engineering** (`src/compute_features.ipynb`): windowing, smoothing, re-centering, rotation, and the 7 kinematic features.
4. **Event detection** (`src/onset_model_train_eval.py`, `src/turning_model_train_eval.py`, `src/turn_10_times_no_align.py`): BiLSTM with Gaussian NLL loss; 10 runs (seeds 42–51), 80/20 split, MAE in ms.
5. **HFS decoding** (`src/hfs_decoding_train_eval.py`): same BiLSTM encoder with a binary classification head (BCE loss) on class-balanced control/HFS trials; reports accuracy overall and per target, within- and between-animal.

## Reproducing the paper

| Paper item | Script / notebook |
|---|---|
| Table 1 (event-detection MAE, within/between, ±aligned) | `onset_model_train_eval.py`, `turning_model_train_eval.py`, `turn_10_times_no_align.py` |
| Fig. 2 (target vectors before/after alignment) | `alignment.ipynb` |
| Fig. 3 (cross-subject scatter) | `onset_model_train_eval.py` / `turning_model_train_eval.py` |
| Figs. 4–5 (within/inter-animal HFS decoding) | `hfs_decoding_train_eval.py` |
| Fig. 6 (architecture schematic) | — (diagram) |

## Decoding (to add)

The public repo currently ships the detection/alignment code. Add the decoding entry point `src/hfs_decoding_train_eval.py` (exported from the project’s decoding notebook), plus the class-balanced control/HFS subset feature files, so Figs. 4–5 are reproducible. It shares the BiLSTM encoder from event detection; only the head (classification vs. regression) and loss differ.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

Released under the **GNU General Public License v3.0** (see `LICENSE`).

## Contact

Corresponding authors: Yifat Prut (yifat.prut@mail.huji.ac.il), Jacob Goldberger (jacob.goldberger@biu.ac.il).
