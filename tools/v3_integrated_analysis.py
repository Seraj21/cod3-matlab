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


PRECODERS = ("ZF", "PZF", "MR")
RETRIES = ["APP50", "CSI-APP50"]
CANDIDATES = ["V3 CNN", "Aligned CNN"]
CONTROLS = ["LS on common packet", "LMMSE", "Aligned LMMSE"]
LABELS = ["LS", "LMMSE", "Matched LMMSE", "CNN", "Perfect CSI at UE"]
STYLES = [
    ("#a0522d", "s", "-"),
    ("#377eb8", "^", "-"),
    ("#5c6d9b", "v", "--"),
    ("#d02e3e", "o", "-"),
    ("#353535", "x", ":"),
]


def profile(rb):
    if rb == 264:
        return {
            "rb": 264, "mhz": 200, "payload_bits": 11880, "coded_bits": 95040,
            "frames_per_batch": 2, "validation_seed": 82000000, "test_seed": 94000000,
            "validation_points": [0, 6, 10, 14, 20], "max_iterations": 200,
        }
    assert rb == 66
    return {
        "rb": 66, "mhz": 50, "payload_bits": PAYLOAD_BITS, "coded_bits": 23760,
        "frames_per_batch": 4, "validation_seed": 76000000, "test_seed": 92000000,
        "validation_points": [0, 4, 6, 8, 10, 12, 16, 20], "max_iterations": 100,
    }


def ceiling(p):
    return 4 * p["payload_bits"] / (SLOT_SECONDS * 1e6)


def load_paired(root, p):
    rows = load_rows(root, ["Precoder", *KEYS], p["payload_bits"], p["max_iterations"])
    assert set(rows.Precoder) == set(PRECODERS)
    return rows


def audit_settings(root, base, batches, points, p):
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
        assert s["FrameCount"] == p["frames_per_batch"] and s["PayloadBits"] == p["payload_bits"], path
        assert s["CodedBits"] == p["coded_bits"] and s["CodeRate"] == .125, path
        assert s["FrameDuration"] == SLOT_SECONDS, path
        assert np.array_equal(s["SNR_dB"], points), path
        assert [cfg[k] for k in ("NumRB", "Users", "UERx", "RFChains", "M", "Qm")] == [
            p["rb"], 4, 4, 8, 512, 9
        ], path
        assert cfg["DelaySpread"] == 1e-6 and cfg["Speed"] == 120, path
        observed.add(key)
    assert observed == expected, f"Missing settings for {expected - observed}"


def freeze(root, output, p):
    rows = load_paired(root, p)
    seed, points = p["validation_seed"], p["validation_points"]
    frames = p["frames_per_batch"]
    audit_settings(root, seed, 4, points, p)
    for precoder in PRECODERS:
        check_design(rows[rows.Precoder == precoder], seeds(seed, 4, frames),
                     points, CONTROLS + CANDIDATES, RETRIES)
    retry_scores = scores(rows, "Retry", RETRIES)
    selected_retry = str(retry_scores.index[0])
    choice = {"Version": 1, "NumRB": p["rb"], "Retry": selected_retry, "CNN": {},
              "SeedBase": seed, "FramesPerArray": 4 * frames, "SNR_dB": points,
              "RetryScores": retry_scores.drop(columns="order").reset_index().to_dict("records"),
              "CNNScores": {}}
    filtered = rows[(rows.Retry == selected_retry) & rows.Estimator.isin(CANDIDATES)]
    for precoder in PRECODERS:
        cnn_scores = scores(filtered[filtered.Precoder == precoder], "Estimator", CANDIDATES)
        choice["CNN"][precoder] = str(cnn_scores.index[0])
        choice["CNNScores"][precoder] = cnn_scores.drop(columns="order").reset_index().to_dict("records")
    write_json(output, choice)
    print(json.dumps(choice, indent=2))


def plot_precoder(precoder, table, methods, out, p):
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10.5})
    for field, slug, title, ylabel in [
        ("PostLDPC_BER", "BER", "BER after LDPC", "Payload bit error rate"),
        ("Goodput_Mbps", "Throughput", "Delivered payload throughput", "Throughput (Mbit/s)"),
    ]:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13.2, 5.55), layout="constrained")
        fig.suptitle(f"{precoder}  |  {title}\n{p['mhz']} MHz  ·  TDL-C  ·  512-QAM  ·  LDPC 1/8  ·  Four users",
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
                ax.set_ylim(.7 / group.PayloadBits.max(), 1.1)
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.text(.02, .028, "▽  Zero observed errors at one-error resolution",
                        transform=ax.transAxes, color="#444444", fontsize=9)
            else:
                ax.set_ylim(0, 1.075 * ceiling(p))
            ax.grid(True, which="major", color="#cdd1d5", alpha=.8)
            ax.grid(True, which="minor", axis="y", color="#e6e8ea", alpha=.6)
            if slug == "BER":
                ax.legend(loc="lower left", bbox_to_anchor=(.02, .11), ncol=2, framealpha=.96)
            else:
                ax.legend(loc="lower right", ncol=2, framealpha=.96)
        rank = PRECODERS.index(precoder) + (1 if slug == "BER" else 4)
        fig.savefig(out / f"{rank:02d}-{precoder}-{slug}.png", dpi=210, facecolor="white")
        plt.close(fig)


def review(root, selection, output, p):
    choice = json.loads(Path(selection).read_text())
    retry = choice["Retry"]
    assert choice["Version"] == 1 and retry in RETRIES and set(choice["CNN"]) == set(PRECODERS)
    assert choice.get("NumRB", 66) == p["rb"]
    rows = load_paired(root, p)
    seed, frames = p["test_seed"], p["frames_per_batch"]
    audit_settings(root, seed, 8, TEST_SNR, p)
    rows.to_csv(Path(output) / "paired_user_blocks.csv", index=False)
    results = []
    summary = {"Selection": choice, "TestSeedBase": seed, "FramesPerArray": 8 * frames,
               "NumRB": p["rb"], "BandwidthMHz": p["mhz"], "SNR_dB": TEST_SNR,
               "ThroughputCeiling_Mbps": ceiling(p), "Precoders": {}}
    figures = Path(output) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for precoder in PRECODERS:
        candidate = choice["CNN"][precoder]
        assert candidate in CANDIDATES
        methods = [*CONTROLS, candidate, "Perfect CSIR"]
        paired = rows[rows.Precoder == precoder]
        check_design(paired, seeds(seed, 8, frames), TEST_SNR, methods, ["None", retry])
        t = aggregate(paired, p["payload_bits"])
        assert t.PayloadBits.eq(32 * frames * p["payload_bits"]).all() and t.Frames.eq(8 * frames).all()
        assert t.Goodput_Mbps.between(0, ceiling(p) + 1e-9).all()
        t.insert(0, "Precoder", precoder)
        results.append(t)
        final = t[t.Retry == retry]
        assert not final.Undetected.any(), f"Undetected payload errors in {precoder}"
        plot_precoder(precoder, final, methods, figures, p)
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
    select.add_argument("--rb", type=int, choices=[66, 264], default=66)
    report = sub.add_parser("review")
    report.add_argument("root")
    report.add_argument("selection")
    report.add_argument("output")
    report.add_argument("--rb", type=int, choices=[66, 264], default=66)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root, args.output, profile(args.rb))
    else:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        review(args.root, args.selection, args.output, profile(args.rb))


if __name__ == "__main__":
    main()
