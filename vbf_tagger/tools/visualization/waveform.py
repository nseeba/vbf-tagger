import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np

hep.style.use(hep.styles.CMS)


def plot_waveform(
        waveform: np.array = None,
        height: np.array = None,
        time: np.array = None,
        figsize: tuple = (16, 9),
        x_label: str = "Time",
        y_label: str = "Amplitude"
):
    fig, ax = plt.subplots(figsize=figsize)
    if waveform is not None:
        time_ = np.arange(len(waveform))
        ax.plot(time_, waveform, label='waveform')
    elif (height is not None) and (time is not None):
        ax.plot(time, height, label='waveform')
    else:
        raise ValueError("Please provide either the waveform or height and time.")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right")
    return fig, ax