import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
hep.style.use(hep.styles.CMS)

import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt

hep.style.use(hep.styles.CMS)


class LossesMultiPlot:
    def __init__(
        self,
        plot_train_losses: bool = False,
        loss_name: str = "CrossEntropy",
        color_mapping: dict = {},
        name_mapping: dict = {},
        x_max: int = -1,
    ):
        self.plot_train_losses = plot_train_losses
        self.loss_name = loss_name
        self.color_mapping = color_mapping
        self.name_mapping = name_mapping
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.x_max = x_max

    def _add_line(self, results: dict, algorithm: str):
        """Adds a line to the plot."""
        self.ax.plot(
            results["val_loss"],
            label=self.name_mapping.get(algorithm, algorithm),
            ls="-",
            color=self.color_mapping.get(algorithm, None),
        )  # Val loss always with solid line
        if self.plot_train_losses:
            self.ax.plot(
                results["train_loss"], ls="--", color=self.color_mapping.get(algorithm, self.ax.lines[-1].get_color())
            )  # Train loss always with dashed line
        self.ax.legend()

    def plot_algorithms(self, results: dict, output_path: str = ""):
        for idx, (algorithm, result) in enumerate(results.items()):
            self._add_line(result, algorithm=algorithm)
        self.ax.set_yscale("log")
        self.ax.set_ylabel(f"{self.loss_name} loss [a.u.]")
        self.ax.set_xlabel("epoch")
        self.ax.set_xlim(0, self.x_max if self.x_max > 0 else None)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


class LossesStackPlot:
    def __init__(
        self,
        loss_name: str = "MSE",
        color_mapping: dict = {},
        name_mapping: dict = {},
        marker_mapping: dict = {},
    ):
        self.loss_name = loss_name
        self.color_mapping = color_mapping
        self.name_mapping = name_mapping
        self.marker_mapping = marker_mapping
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def _add_line(self, results: dict, algorithm: str, y: int):
        self.ax.errorbar(
            np.mean(results["best_losses"]),
            y,
            xerr=np.std(results["best_losses"]),
            label=self.name_mapping.get(algorithm, algorithm),
            color=self.color_mapping.get(algorithm, None),
            marker=self.marker_mapping.get(algorithm, "o"),
            ls="",
            ms=10,
            capsize=5,
        )

    def plot_algorithms(self, results: dict, output_path: str = ""):
        yticklabels = []
        for idx, (algorithm, result) in enumerate(results.items()):
            yticklabels.append(self.name_mapping.get(algorithm, algorithm))
            self._add_line(result, algorithm=algorithm, y=idx)
        self.ax.set_xlabel(f"{self.loss_name} loss [a.u.]")
        self.ax.set_yticks(np.arange(len(yticklabels)))
        self.ax.set_yticklabels(yticklabels)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


class LossesStackPlot2:
    def __init__(
        self,
        loss_name: str = "MSE",
        baseline_algo: str = None,
        color_mapping: dict = {},
        name_mapping: dict = {},
        marker_mapping: dict = {},
    ):
        """Plots losses relative to one indicated by the 'baseline_algo'.
        If not set, uses the worst performing algo"""
        self.loss_name = loss_name
        self.color_mapping = color_mapping
        self.baseline_algo = baseline_algo
        self.name_mapping = name_mapping
        self.marker_mapping = marker_mapping
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def _add_line(self, results: dict, algorithm: str, y: int, baseline_value):
        self.ax.errorbar(
            results["mean"] / baseline_value,
            y,
            xerr=results["err"] / baseline_value,
            label=self.name_mapping.get(algorithm, algorithm),
            color=self.color_mapping.get(algorithm, None),
            marker=self.marker_mapping.get(algorithm, "o"),
            ls="",
            ms=10,
            capsize=5,
        )

    def plot_algorithms(self, results: dict, output_path: str = ""):
        yticklabels = []
        processed_results = {}
        for idx, (algorithm, result) in enumerate(results.items()):
            yticklabels.append(self.name_mapping.get(algorithm, algorithm))
            processed_results[algorithm] = {
                "mean": np.mean(result["best_losses"]),
                "err": np.std(result["best_losses"]),
            }
        max_mean = np.max([result["mean"] for _, result in processed_results.items()])
        baseline_value = processed_results[algorithm]["mean"] if self.baseline_algo is not None else max_mean
        for idx, (algorithm, result) in enumerate(processed_results.items()):
            self._add_line(result, algorithm=algorithm, y=idx, baseline_value=baseline_value)
        self.ax.axvline(1, ls="--", color="k")
        self.ax.set_xlabel(f"{self.loss_name} loss [a.u.]")
        self.ax.set_yticks(np.arange(len(yticklabels)))
        self.ax.set_yticklabels(yticklabels)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


# def plot_loss_evolution(
#         val_loss: np.array,
#         train_loss: np.array,
#         output_path: str = "",
#         loss_name: str = "MSE"
# ):
#     """ Plots the evolution of train and validation loss.

#     Parameters:
#         val_loss : np.array
#             Validation losses for the epochs
#         train_loss : np.array
#             Training losses for the epochs
#         output_path : str
#             [default: ''] Path where figure will be saved. If empty string, figure will not be saved
#         loss_name : str
#             [default: "MSE"] Loss function name used for the training

#     Returns:
#         None
#     """
#     # if multirun case?
#     plt.plot(val_loss, label="val_loss", color='k')
#     if train_loss is not None:
#         plt.plot(train_loss, label="train_loss", ls="--", color='k')
#     plt.grid()
#     plt.yscale('log')
#     plt.ylabel(f'{loss_name} loss [a.u.]')
#     plt.xlabel('epoch')
#     plt.xlim(0, len(val_loss))
#     plt.legend()
#     plt.savefig(output_path)
#     if output_path != '':
#         plt.savefig(output_path, bbox_inches='tight')
#     plt.close("all")