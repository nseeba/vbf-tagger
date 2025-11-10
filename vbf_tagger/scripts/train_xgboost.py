import hydra
from omegaconf import DictConfig
from vbf_tagger.tools.data.flatpair_dataloader import FlatPairDataModule
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def train_xgboost(cfg: DictConfig):
    print("Training XGBoost VBF Pair Classifier")

    # Load flat data
    dm = FlatPairDataModule(cfg)
    dm.setup(stage="fit")

    X_train = dm.train_dataset.tensors[0].numpy()
    y_train = dm.train_dataset.tensors[1].numpy()

    X_val = dm.val_dataset.tensors[0].numpy()
    y_val = dm.val_dataset.tensors[1].numpy()

    print(f" Train: {X_train.shape}, Val: {X_val.shape}")

    w_train = np.ones_like(y_train, dtype=float)
    N_sig = (y_train == 1).sum()
    N_bkg = (y_train == 0).sum()
    w_sig = 100000 / N_sig
    w_bkg = 100000 / N_bkg
    w_train[y_train == 1] = w_sig
    w_train[y_train == 0] = w_bkg

    w_val   = np.ones_like(y_val, dtype=float)
    N_sig_val = (y_val == 1).sum()
    N_bkg_val = (y_val == 0).sum()
    w_sig_val = 100000 / N_sig_val
    w_bkg_val = 100000 / N_bkg_val
    w_val[y_val == 1] = w_sig_val
    w_val[y_val == 0] = w_bkg_val

    # Build XGBoost DMatrix
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dval   = xgb.DMatrix(X_val,   label=y_val)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval   = xgb.DMatrix(X_val,   label=y_val,   weight=w_val)

    # XGB params (starting point)
    params = {
        "max_depth": 4,
        "eta": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "nthread": 8,
        "tree_method": "hist",
        "lambda": 1.2,
        "alpha": 0.2,
        "gamma": 0.5,
        "min_child_weight": 3 
    }

    evals = [(dtrain, "train"), (dval, "val")]

    # Train with early stopping
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=evals,
        early_stopping_rounds=80,
        verbose_eval=50
    )

    # Evaluate
    preds_val = bst.predict(dval)
    auc_val = roc_auc_score(y_val, preds_val)
    print(f" Validation AUC = {auc_val:.4f}")

    # Save model
    out = "XGBoost/xgboost_vbf_251107.json"
    bst.save_model(out)
    print(f" Saved model to {out}")

if __name__ == "__main__":
    train_xgboost()
