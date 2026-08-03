# README Models

This folder does not include the trained model checkpoint directly, `champion.joblib` is around 200 KB, small enough for Git, but is excluded anyway to keep model artifacts reproducible from the notebook rather than silently drifting from it.

To get the files:

1. Run `notebooks/02_model_training_baseline.ipynb` from top to bottom. It trains the model on `data/feature_table_fp.csv` and saves `champion.joblib`, `champion_meta.json`, `model_v0.joblib`, and `registry.json` into this folder.
2. If you only need to run the API and do not want to retrain anything, ask a team member for the current `champion.joblib` and `champion_meta.json` and place them directly in this folder.

Expected files after either path: `champion.joblib`, `champion_meta.json`, `model_v0.joblib`, `registry.json`.
