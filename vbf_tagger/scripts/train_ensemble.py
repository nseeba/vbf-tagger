import os
import hydra
import joblib
import numpy as np
import torch
import xgboost as xgb

from hydra import initialize, compose
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from omegaconf import DictConfig

from vbf_tagger.models.MLPClassifier import MLPClassifier as MLPLevel0
from vbf_tagger.tools.data.flatpair_dataloader import FlatPairDataModule


def get_mlp_predictions(model, loader):
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            probs = torch.sigmoid(logits).squeeze()
            preds.append(probs.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)

@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_ensemble(cfg: DictConfig):
    # DataModule
    dm = FlatPairDataModule(cfg)
    dm.setup(stage="fit")
    dm.setup(stage="test")

    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()


    # Load Level-0 MLP
    ckpt_path = cfg.ensemble.mlp_checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hyperparams = checkpoint["hyper_parameters"]
    mlp_model = MLPLevel0(**hyperparams)
    mlp_model.load_state_dict(checkpoint["state_dict"])
    mlp_model.eval()

    print("Loaded Level-0 MLP.")


    # 3. Load Level-0 XGBoost
    xgb_path = cfg.ensemble.xgb_model
    bst = xgb.Booster()
    bst.load_model(xgb_path)

    print("Loaded Level-0 XGBoost.")


    # Get predictions for both models
    # MLP predictions
    mlp_train_pred, y_train = get_mlp_predictions(mlp_model, train_loader)
    mlp_test_pred, y_test = get_mlp_predictions(mlp_model, test_loader)

    # XGB predictions
    X_train = dm.train_dataset.tensors[0].numpy()
    X_test  = dm.test_dataset.tensors[0].numpy()

    dtrain = xgb.DMatrix(X_train)
    dtest  = xgb.DMatrix(X_test)

    xgb_train_pred = bst.predict(dtrain)
    xgb_test_pred  = bst.predict(dtest)


    # 5. Build meta features
    X_meta_train = np.vstack([mlp_train_pred, xgb_train_pred]).T
    X_meta_test  = np.vstack([mlp_test_pred,  xgb_test_pred]).T

    print("Meta features shape:")
    print("Train:", X_meta_train.shape)
    print("Test :", X_meta_test.shape)


    # Define Level-1 candidate models
    meta_models = {
            "logreg": LogisticRegression(max_iter=5000),
            "tiny_xgb": XGBClassifier(max_depth=2, n_estimators=50,
                                    learning_rate=0.05, tree_method="hist"),
            "tiny_mlp": MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000)
        }

    out_dir = cfg.ensemble.output_dir
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    # Train each meta model
    for name, model in meta_models.items():
        print(f"\nTraining meta-model: {name}")

        model.fit(X_meta_train, y_train)
        pred_test = model.predict_proba(X_meta_test)[:, 1]

        auc_test = roc_auc_score(y_test, pred_test)
        results[name] = auc_test

        # Save model
        save_path = os.path.join(out_dir, f"meta_{name}.pkl")
        joblib.dump(model, save_path)

        print(f"Saved {name} → {save_path}")
        print(f"Test AUC = {auc_test:.6f}")


    # Report best model
    best = max(results, key=results.get)
    print("\n========== ENSEMBLE RESULTS ==========")
    for name, aucv in results.items():
        print(f"{name:10s}: AUC = {aucv:.6f}")

    print("\nBest model:", best)
    print("Stored in:", out_dir)


if __name__ == "__main__":
    train_ensemble()