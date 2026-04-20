# KickCast — 2026 FIFA World Cup Prediction

An end-to-end machine learning pipeline that predicts international football match outcomes (Home Win / Draw / Away Win) and uses the calibrated probabilities to simulate the 2026 FIFA World Cup via Monte Carlo.

**EECE 5644 — Introduction to Machine Learning** | Spring 2026
Karim Semaan & Ramzi Zeineddine

## Reproduce the results

```bash
git clone https://github.com/karimsemaan/KickCaster.git
cd KickCaster
pip install -r requirements.txt
jupyter notebook
```

Then open the notebooks in order:

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `notebooks/01_eda.ipynb` | Exploratory data analysis |
| 2 | `notebooks/02_model_training.ipynb` | Trains 7 classifiers + tuning |
| 3 | `notebooks/03_model_evaluation.ipynb` | Evaluates models on the 2022 World Cup holdout |
| 4 | `notebooks/04_world_cup_simulation.ipynb` | 10,000-iteration Monte Carlo simulation of the 2026 World Cup |
| 5 | `notebooks/05_shap_analysis.ipynb` | SHAP feature-importance analysis |

The processed feature matrix (`data/processed/`) and all trained models (`outputs/model_artifacts/`) are included in the repository, so notebooks 3–5 run in a few minutes without any retraining.

## Rebuild the data from scratch (optional)

```bash
python data/scripts/01_download_data.py
python data/scripts/02_build_features.py
python data/scripts/03_create_splits.py
```

Requires a Kaggle API key at `~/.kaggle/kaggle.json` (https://www.kaggle.com/docs/api).

## Repository layout

```
KickCaster/
├── data/
│   ├── raw/world_cup_2026/      # 2026 fixtures, groups, bracket, venues
│   ├── processed/               # feature matrix and train/val/test splits
│   └── scripts/                 # data collection and feature-engineering scripts
├── notebooks/                   # analysis notebooks (run in order)
├── src/                         # reusable Python modules
└── outputs/
    ├── figures/                 # generated figures
    ├── model_artifacts/         # trained models (.joblib)
    └── simulation_results/      # 2026 World Cup simulation output
```

## Task

- **Input:** 25+ delta features (home minus away) per match — Elo rating, FIFA ranking, rolling form, Transfermarkt squad value, injury burden, manager experience, head-to-head history, World Cup history, and match context.
- **Target:** three-class match outcome (Home Win / Draw / Away Win).
- **Training window:** 2004-01-01 through 2019-12-31.
- **Validation:** 2020-01-01 through 2022-11-19.
- **Test:** 2022 FIFA World Cup.
- **Models:** Logistic Regression, KNN, Random Forest, XGBoost, HistGradientBoosting, SVM (RBF), Stacking Ensemble.
- **Simulation:** 10,000 Monte Carlo iterations over the full 2026 World Cup fixture list (group stage + knockout).

## Data sources

| Source | Content |
|--------|---------|
| [martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 50k+ international matches |
| [eloratings.net](https://www.eloratings.net/) | Live Elo ratings and match-by-match histories |
| [cashncarry/fifaworldranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Historical FIFA rankings |
| [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Transfermarkt player market values, appearances, managers |
| [salimt/football-datasets](https://github.com/salimt/football-datasets) | Historical player injuries |
| [piterfm/fifa-football-world-cup](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup) | World Cup match history 1930–2022 |

## License

MIT
