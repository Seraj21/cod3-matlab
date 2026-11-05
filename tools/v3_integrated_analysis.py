"""Freeze and audit a common receiver comparison across MR, ZF and PZF."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import pandas as pd
from scipy.io import loadmat

from v4_pzf_analysis import (
    KEYS, PAYLOAD_BITS, SLOT_SECONDS, TEST_SNR, aggregate, check_design,
    load_rows, paired_interval, scores, seeds, write_json,
)


PRECODERS = ("MR", "ZF", "PZF")
RETRIES = ["APP50", "CSI-APP50"]
CANDIDATES = ["V3 CNN", "Aligned CNN"]
CONTROLS = ["LS on common packet", "LMMSE", "Aligned LMMSE"]
VALIDATION_POINTS = [0, 4, 6, 8, 10, 12, 16, 20]
LABELS = ["LS", "LMMSE", "Matched LMMSE", "CNN", "Perfect CSI at UE"]
STYLES = [
    ("#a0522d", "s", "-"),
    ("#377eb8", "^", "-"),
    ("#5c6d9b", "v", "--"),
    ("#d02e3e", "o", "-"),
    ("#353535", "x", ":"),
]
CEILING = 4 * PAYLOAD_BITS / (SLOT_SECONDS * 1e6)


def load_paired(root):
    rows = load_rows(root, ["Precoder", *KEYS])
    assert set(rows.Precoder) == set(PRECODERS)
    return rows


def audit_settings(root, base, batches, points):
    expected = {
        (scheme, bs, base + batch * 100000)
        for scheme in PRECODERS for bs in (32, 64) for batch in range(batches)
    }
    observed = set()
    for path in sorted(Path(root).rglob("pzf_settings.mat")):
        s = loadmat(path, simplify_cells=True)["settings"]
        cfg = s["Config"]
        scheme = path.parent.name.split("-")[-3]
        key = (scheme, int(cfg["Antennas"]), int(s["SeedOffset"]))
        assert key in expected and key not in observed, path
        assert s["FrameCount"] == 4 and s["PayloadBits"] == PAYLOAD_BITS, path
        assert s["CodedBits"] == 23760 and s["CodeRate"] == .125, path
        assert s["FrameDuration"] == SLOT_SECONDS, path
        assert np.array_equal(s["SNR_dB"], points), path
        assert [cfg[k] for k in ("NumRB", "Users", "UERx", "RFChains", "M", "Qm")] == [
            66, 4, 4, 8, 512, 9
        ], path
        assert cfg["DelaySpread"] == 1e-6 and cfg["Speed"] == 120, path
        observed.add(key)
    assert observed == expected, f"Missing settings for {expected - observed}"


def freeze(root, output):
    rows = load_paired(root)
    audit_settings(root, 76000000, 4, VALIDATION_POINTS)
    for precoder in PRECODERS:
        check_design(rows[rows.Precoder == precoder], seeds(76000000, 4),
                     VALIDATION_POINTS, CONTROLS + CANDIDATES, RETRIES)
    retry_scores = scores(rows, "Retry", RETRIES)
    selected_retry = str(retry_scores.index[0])
    choice = {"Version": 1, "Retry": selected_retry, "CNN": {},
              "SeedBase": 76000000, "FramesPerArray": 16, "SNR_dB": VALIDATION_POINTS,
              "RetryScores": retry_scores.drop(columns="order").reset_index().to_dict("records"),
              "CNNScores": {}}
    filtered = rows[(rows.Retry == selected_retry) & rows.Estimator.isin(CANDIDATES)]
    for precoder in PRECODERS:
        cnn_scores = scores(filtered[filtered.Precoder == precoder], "Estimator", CANDIDATES)
        choice["CNN"][precoder] = str(cnn_scores.index[0])
        choice["CNNScores"][precoder] = cnn_scores.drop(columns="order").reset_index().to_dict("records")
    write_json(output, choice)
    print(json.dumps(choice, indent=2))


def plot_precoder(precoder, table, methods, out):
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10.5})
    for field, slug, title, ylabel in [
        ("PostLDPC_BER", "BER", "BER after LDPC", "Payload bit error rate"),
        ("Goodput_Mbps", "Throughput", "Delivered payload throughput", "Throughput (Mbit/s)"),
    ]:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13.2, 5.55), layout="constrained")
        fig.suptitle(f"{precoder}  |  {title}\nTDL-C  ·  512-QAM  ·  LDPC 1/8  ·  Four users",
                     fontsize=16, fontweight="semibold")
        for ax, bs in zip(axes, (32, 64)):
            group = table[table.BS == bs]
            for method, label, (color, marker, linestyle) in zip(methods, LABELS, STYLES):
                s = group[group.Estimator == method].sort_values("SNR_dB")
                x, y = s.SNR_dB.to_numpy(), s[field].to_numpy()
                if slug == "BER":
                    zero = y == 0
                    if zero.any():
                        ax.plot(x, np.where(zero, 1 / s.PayloadBits.to_numpy(), np.nan),
                                marker="v", markersize=5 if label == "CNN" else 7,
                                markerfacecolor="none", color=color, linestyle=":",
                                linewidth=1, zorder=7 if label == "CNN" else 6)
                    y = np.where(zero, np.nan, y)
                ax.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.75, markersize=3.6, label=label)
            ax.set_title(f"{bs} BS antennas", fontsize=13)
            ax.set_xlim(0, 20)
            ax.set_xticks(range(0, 21, 2))
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="y", labelleft=True)
            if slug == "BER":
                ax.set_yscale("log")
                ax.set_ylim(1e-6, 1.1)
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.text(.02, .028, "▽  Zero observed errors at one-error resolution",
                        transform=ax.transAxes, color="#444444", fontsize=9)
            else:
                ax.set_ylim(0, 51)
            ax.grid(True, which="major", color="#cdd1d5", alpha=.8)
            ax.grid(True, which="minor", axis="y", color="#e6e8ea", alpha=.6)
            ax.legend(loc="upper right" if slug == "BER" else "lower right",
                      ncol=2, framealpha=.96)
        fig.savefig(out / f"{precoder}-{slug}.png", dpi=210, facecolor="white")
        plt.close(fig)


def review(root, selection, output):
    choice = json.loads(Path(selection).read_text())
    retry = choice["Retry"]
    assert choice["Version"] == 1 and retry in RETRIES and set(choice["CNN"]) == set(PRECODERS)
    rows = load_paired(root)
    audit_settings(root, 92000000, 8, TEST_SNR)
    rows.to_csv(Path(output) / "paired_user_blocks.csv", index=False)
    results = []
    summary = {"Selection": choice, "TestSeedBase": 92000000, "FramesPerArray": 32,
               "SNR_dB": TEST_SNR, "ThroughputCeiling_Mbps": CEILING, "Precoders": {}}
    figures = Path(output) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for precoder in PRECODERS:
        candidate = choice["CNN"][precoder]
        assert candidate in CANDIDATES
        methods = [*CONTROLS, candidate, "Perfect CSIR"]
        paired = rows[rows.Precoder == precoder]
        check_design(paired, seeds(92000000, 8), TEST_SNR, methods, ["None", retry])
        t = aggregate(paired)
        assert t.PayloadBits.eq(379904).all() and t.Frames.eq(32).all()
        assert t.Goodput_Mbps.between(0, CEILING + 1e-9).all()
        t.insert(0, "Precoder", precoder)
        results.append(t)
        final = t[t.Retry == retry]
        assert not final.Undetected.any(), f"Undetected payload errors in {precoder}"
        plot_precoder(precoder, final, methods, figures)
        ber = final.pivot(index=["BS", "SNR_dB"], columns="Estimator", values="PostLDPC_BER")
        paired_final = paired[paired.Retry == retry]
        average = final.groupby("Estimator")[["PostLDPC_BER", "Goodput_Mbps"]].mean()
        summary["Precoders"][precoder] = {
            "Average": average.to_dict("index"),
            "CNNWinsAgainstMatchedLMMSE": int((ber[candidate] < ber["Aligned LMMSE"]).sum()),
            "CNNTiesAgainstMatchedLMMSE": int((ber[candidate] == ber["Aligned LMMSE"]).sum()),
            "PairedMeanBERGainCI95": paired_interval(paired_final, candidate, "Aligned LMMSE"),
            "At20dB": final[final.SNR_dB == 20].to_dict("records"),
        }
        print(precoder, "CNN:", candidate, "mean over 62 paired SNR/size points")
        print(average.to_string())
        print(final[final.SNR_dB == 20][
            ["BS", "Estimator", "PayloadBitErrors", "CorrectTransportBlocks", "PostLDPC_BER", "Goodput_Mbps"]
        ].to_string(index=False))
    pd.concat(results, ignore_index=True).to_csv(Path(output) / "metrics.csv", index=False)
    write_json(Path(output) / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    select = sub.add_parser("freeze")
    select.add_argument("root")
    select.add_argument("output")
    report = sub.add_parser("review")
    report.add_argument("root")
    report.add_argument("selection")
    report.add_argument("output")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root, args.output)
    else:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        review(args.root, args.selection, args.output)


if __name__ == "__main__":
    main()
