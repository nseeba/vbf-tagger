import numpy as np
import awkward as ak
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d


def thresholded_roc_curve(truth, predictions, n_classifier_cuts=100) -> dict:
    print("Calculating ROC curve...")
    thresholds = np.linspace(start=0, stop=1, num=n_classifier_cuts + 1)
    tpr = []
    fpr = []
    truth = ak.to_numpy(truth)
    predictions = ak.to_numpy(predictions)
    sig_mask = truth == 1
    bkg_mask = truth == 0
    for threshold in thresholds:
        tpr.append(np.sum(predictions[sig_mask] >= threshold) / np.sum(sig_mask))
        fpr.append(np.sum(predictions[bkg_mask] >= threshold) / np.sum(bkg_mask))
    auc = roc_auc_score(truth, predictions)
    roc_results = {
        "FPR": np.array(fpr),
        "TPR": np.array(tpr),
        "AUC": auc,
        "thresholds": thresholds,
        "pred": predictions,
        "true": truth,
    }
    return roc_results


def get_metric_cut(results, at_fakerate: float = -1, at_efficiency: float = -1):
    print("Calculating metric cut...")
    # TODO: eff/fakerate at X for all energies in avarage
    if (at_fakerate != -1) and (at_efficiency != -1):
        raise ValueError("You can only choose fixed value for either `at_efficiency` or `at_fakerate`")
    elif (at_fakerate == -1) and (at_efficiency == -1):
        raise ValueError("Please choose a fixed value for either `at_efficiency` or `at_fakerate`")
    if at_fakerate != -1:
        nearest_idx = np.argmin(np.abs(results["FPR"] - at_fakerate))
        efficiency = results["TPR"][nearest_idx]
        print(f"Using threshold for {at_fakerate:.2f} fakerate, which corresponds to {efficiency:.2f} efficiency")
    if at_efficiency != -1:
        nearest_idx = np.argmin(np.abs(results["TPR"] - at_efficiency))
        fakerate = results["FPR"][nearest_idx]
        print(f"Using threshold for {at_efficiency:.2f} efficiency, which corresponds to {fakerate:.2f} fakerate")
    return results["thresholds"][nearest_idx], nearest_idx


def calculate_metrics(
    truth, predictions, at_fakerate: float = -1, at_efficiency: float = -1
) -> dict:
    """
    Calculate ROC curve and metrics based on the truth and predictions.

    Parameters:
        truth (np.array): True labels (0 or 1).
        predictions (np.array): Predicted probabilities or scores.
        at_fakerate (float): Desired false positive rate to calculate the threshold.
        at_efficiency (float): Desired true positive rate to calculate the threshold.

    Returns:
        dict: Contains ROC results and the chosen threshold.
    """
    print("Calculating metrics...")
    roc_results = thresholded_roc_curve(truth, predictions)
    update = {}
    if at_fakerate != -1:
        fr_threshold, idx = get_metric_cut(roc_results, at_fakerate=at_fakerate)
        update["fr_threshold"] = fr_threshold
        update["fr_cut_idx"] = idx
    if at_efficiency != -1:
        eff_threshold, idx = get_metric_cut(roc_results, at_efficiency=at_efficiency)
        update["eff_threshold"] = eff_threshold
        update["eff_cut_idx"] = idx
    results = dict(update, **roc_results)
    return results


# def get_per_energy_metrics(
#     results: dict, at_fakerate: float = -1, at_efficiency: float = -1, signal: str = "both"
# ) -> dict:
#     """
#     Calculate metrics for each energy in the results dictionary.

#     Parameters:
#         results (dict): Dictionary containing truth and predictions for each energy.
#         at_fakerate (float): Desired false positive rate to calculate the threshold.
#         at_efficiency (float): Desired true positive rate to calculate the threshold.

#     Returns:
#         dict: Contains metrics for each energy.
#     """
#     print("Calculating per energy metrics...")
#     all_results = {}
#     for pid, pid_results in results.items():
#         all_results[pid] = {}
#         for energy, data in pid_results.items():
#             true = ak.flatten(data["true"])
#             target_mask = (true != -1) & (true != -999)
#             true = true[target_mask]
#             pred = ak.flatten(data["pred"])
#             pred = pred[target_mask]
#             all_results[pid][energy] = calculate_metrics(
#                 truth=true, predictions=pred, at_fakerate=at_fakerate, at_efficiency=at_efficiency, signal=signal
#             )
#         all_results[pid]["global"] = calculate_global_metrics(all_results[pid])
#     return all_results


# def calculate_global_auc(all_aucs: list) -> dict:
#     """
#     Calculate the global AUC from a list of AUCs.

#     Parameters:
#         all_aucs (list): List of AUC values for each energy.

#     Returns:
#         tuple: Contains the mean and standard deviation of the AUCs.
#     """
#     print("Calculating global AUC...")
#     if len(all_aucs) == 0:
#         return None, None
#     mean_auc = np.mean(all_aucs)
#     std_auc = np.std(all_aucs)
#     global_aucs = {
#         "mean": mean_auc,
#         "std": std_auc,
#     }
#     return global_aucs


# def calculate_global_roc(all_rocs: list) -> dict:
#     """
#     Calculate the global ROC curve from a list of ROC results.

#     Parameters:
#         all_rocs (list): List of ROC results for each energy.

#     Returns:
#         dict: Contains the mean and standard deviation of FPR and TPR.
#     """
#     print("Calculating global ROC curve...")
#     x_common = np.linspace(0, 1, 101)
#     y_values = []
#     for roc in all_rocs:
#         y_interp = interp1d(roc["FPR"], roc["TPR"], kind="linear", fill_value="extrapolate")(x_common)
#         y_values.append(y_interp)
#     y_stack = np.stack(y_values, axis=0)
#     y_mean_ft = np.mean(y_stack, axis=0)
#     y_std_ft = np.std(y_stack, axis=0)
#     global_roc = {
#         "FPRs": x_common,
#         "avg_TPRs": y_mean_ft,
#         "std_TPRs": y_std_ft,
#     }
#     return global_roc


# def calculate_global_metrics(pid_results: dict) -> dict:
#     """
#     Calculate global metrics across all energies.

#     Parameters:
#         all_results (dict): Dictionary containing truth and predictions for each energy.

#     Returns:
#         dict: Contains global metrics.
#     """
#     print("Calculating global metrics...")
#     all_results = {}
#     all_pred = []
#     all_true = []
#     all_rocs = []
#     all_aucs = []
#     all_results = {"FPRs": [], "TPRs": [], "energies": []}
#     for energy, energy_results in pid_results.items():
#         all_pred.extend(energy_results["pred"])
#         all_true.extend(energy_results["true"])
#         all_rocs.append({"FPR": energy_results["FPR"], "TPR": energy_results["TPR"]})
#         all_results["FPRs"].append(energy_results["FPR"][energy_results["eff_cut_idx"]])
#         all_results["TPRs"].append(energy_results["TPR"][energy_results["fr_cut_idx"]])
#         all_aucs.append(energy_results["AUC"])
#         all_results["energies"].append(energy)
#     all_results["AUC_mean"] = roc_auc_score(all_true, all_pred)
#     all_results["AUC"] = calculate_global_auc(all_aucs)
#     all_results["ROC"] = calculate_global_roc(all_rocs)
#     return all_results