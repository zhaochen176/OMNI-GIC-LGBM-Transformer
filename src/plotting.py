from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter


def plot_series(datetime, y_true, y_pred, out_path: Path, title: str = "GIC Prediction"):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(datetime, y_true, label="Measured", linewidth=1.0)
    plt.plot(datetime, y_pred, label="Predicted", linewidth=1.0)

    ax = plt.gca()
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=30)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("GIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
