import argparse
from pathlib import Path
import pandas as pd

from src.plotting import plot_driver_fivepanel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV file path")
    p.add_argument("--datetime_col", default="date", help="Datetime column name")
    p.add_argument("--out", default="outputs/figures/Five_Subplots_HQ_Hourly_Xaxis_Full.png")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df[args.datetime_col] = pd.to_datetime(df[args.datetime_col])

    cols = ["Akasofu", "SYM/H", "SYM/H_diff1", "SYM/H_diff2", "GIC"]
    labels = ["Akasofu ε (W)", "SYM-H(nT)", "d(SYM-H)/dt(nT/s)", "d²(SYM-H)/dt²(nT/s²)", "GIC(A)"]

    plot_driver_fivepanel(
        df=df,
        datetime_col=args.datetime_col,
        cols=cols,
        labels=labels,
        out_path=Path(args.out),
        title="Physical driving characteristics: Akasofu(W), SYM-H(nT)",
        xlabel="Time (UTC)",
        dpi=300,
    )


if __name__ == "__main__":
    main()
