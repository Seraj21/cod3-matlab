"""Validate paired PZF counts and summarize the independently frozen comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import pandas as pd


RETRIES = ["None", "LBP50", "APP20", "CSI-APP20", "CSI-APP50"]
ORIGINAL = ["LS", "LMMSE", "V3 CNN", "Perfect CSI"]
CNN_CANDIDATES = ["V3 CNN", "Matched CNN", "Aligned CNN"]
VALIDATION = ORIGINAL[:3] + ["Matched LMMSE", "Matched CNN", "Aligned CNN"]
CALIBRATION_SNR = [0, 4, 8, 12, 16, 20]
VALIDATION_SNR = list(range(0, 21, 2))
TEST_SNR = list(np.arange(0, 10.5, 0.5)) + list(range(11, 21))
SLOT_SECONDS = 0.00025
PAYLOAD_BITS = 2968
KEYS = ["BS", "ChannelSeed", "SNR_dB", "Estimator", "UE"]


def load_rows(root):
    files = sorted(Path(root).rglob("pzf_slots.csv"))
    assert files, f"No measurements in {root}"
    rows = pd.concat(
        [pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True
    )
    for key in ["CRCFailed", "BaselineCRCFailed", "SRSBeamAccepted"]:
        mapped = rows[key].astype(str).str.lower().map(
            {"0": False, "1": True, "false": False, "true": True}
        )
        assert mapped.notna().all(), f"Invalid boolean field {key}"
        rows[key] = mapped
    rows["Correct"] = (~rows.CRCFailed) & (rows.Errors == 0)
    rows["BaselineCorrect"] = (~rows.BaselineCRCFailed) & (rows.BaselineErrors == 0)
    rows["Undetected"] = (~rows.CRCFailed) & (rows.Errors != 0)
    assert rows.PayloadBits.eq(PAYLOAD_BITS).all()
    assert rows.Errors.between(0, PAYLOAD_BITS).all()
    assert rows.BaselineErrors.between(0, PAYLOAD_BITS).all()
    assert rows.ExtraIterations.between(0, 100).all()
    assert not rows.duplicated(KEYS + ["Retry"]).any()
    unchanged = (~rows.BaselineCRCFailed) | (rows.Retry == "None") | (
        rows.Estimator == "Perfect CSI"
    ) | rows.CRCFailed
    assert rows.loc[unchanged, "Errors"].eq(rows.loc[unchanged, "BaselineErrors"]).all()
    assert rows.loc[unchanged, "CRCFailed"].eq(
        rows.loc[unchanged, "BaselineCRCFailed"]
    ).all()
    assert not rows.loc[~rows.BaselineCRCFailed, "ExtraIterations"].any()
    baseline = rows.groupby(KEYS)[["BaselineErrors", "BaselineCRCFailed"]].nunique()
    assert baseline.eq(1).all().all()
    return rows


def seeds(base, batches, per_batch=4):
    return [base + batch * 100000 + i for batch in range(batches) for i in range(1, per_batch + 1)]


def check_design(rows, frames, snr, methods, modes):
    expected = pd.MultiIndex.from_product(
        [[32, 64], frames, snr, methods, [1, 2, 3, 4], modes],
        names=KEYS + ["Retry"],
    )
    actual = pd.MultiIndex.from_frame(rows[KEYS + ["Retry"]])
    assert len(actual) == len(expected) and actual.sort_values().equals(expected.sort_values())


def scores(rows, column, order):
    groups = rows.groupby(column).agg(
        errors=("Errors", "sum"), bits=("PayloadBits", "sum"),
        correct=("Correct", "sum"), extra_iterations=("ExtraIterations", "sum"),
    )
    groups = groups.loc[order].copy()
    groups["ber"] = groups.errors / groups.bits
    groups["order"] = np.arange(len(groups))
    return groups.sort_values(["errors", "correct", "order"], ascending=[True, False, True])


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_receiver(root, out):
    rows = load_rows(root)
    check_design(rows, seeds(71000000, 2), CALIBRATION_SNR, ORIGINAL[:3], RETRIES)
    score = scores(rows, "Retry", RETRIES)
    choice = str(score.index[0])
    write_json(out, {
        "retry": choice, "frames_per_array": 8, "SNR_dB": CALIBRATION_SNR,
        "selection": score.drop(columns="order").reset_index().to_dict("records"),
    })
    print(score.to_string())
    print("Frozen common retry:", choice)


def select_cnn(root, receiver_choice, out):
    receiver = json.loads(Path(receiver_choice).read_text())
    rows = load_rows(root)
    check_design(rows, seeds(74000000, 4), VALIDATION_SNR, VALIDATION, [receiver["retry"]])
    score = scores(rows[rows.Estimator.isin(CNN_CANDIDATES)], "Estimator", CNN_CANDIDATES)
    controls = scores(rows, "Estimator", VALIDATION).drop(columns="order")
    choice = str(score.index[0])
    write_json(out, {
        "cnn": choice, "frames_per_array": 16, "SNR_dB": VALIDATION_SNR,
        "common_receiver": receiver,
        "selection": controls.reset_index().to_dict("records"),
    })
    print(controls.to_string())
    print("Frozen CNN:", choice)


def replay(new_root, old_root):
    rows = load_rows(new_root)
    check_design(rows, [50000001], [0, 10, 20], ORIGINAL, ["None"])
    old = []
    for bs in [32, 64]:
        path = Path(old_root) / f"v3-final-evaluation-{bs}-0" / "mu_slot_errors.csv"
        data = pd.read_csv(path, keep_default_na=False)
        data = data[(data.Packet == 1) & (data.Precoder == "PZF") &
                    data.SNR_dB.isin([0, 10, 20])].copy()
        data["BS"] = bs
        data["Estimator"] = data.Estimator.replace({"CNN": "V3 CNN"})
        old.append(data.rename(columns={"User": "UE", "PayloadBitErrors": "Errors"}))
    keys = ["BS", "SNR_dB", "Estimator", "UE"]
    fields = keys + ["Errors", "CRCFailed", "SRSBeamAccepted"]
    pd.testing.assert_frame_equal(
        rows[fields].sort_values(keys).reset_index(drop=True),
        pd.concat(old)[fields].sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )
    print(f"Exact V3 replay: {len(rows)} user-block measurements match.")


def aggregate(rows):
    groups = rows.groupby(["BS", "Estimator", "Retry", "SNR_dB"], sort=True)
    table = groups.agg(
        PayloadBits=("PayloadBits", "sum"), PayloadBitErrors=("Errors", "sum"),
        BaselineBitErrors=("BaselineErrors", "sum"), CorrectTransportBlocks=("Correct", "sum"),
        BaselineCorrectBlocks=("BaselineCorrect", "sum"), Undetected=("Undetected", "sum"),
        ExtraIterations=("ExtraIterations", "sum"), Frames=("ChannelSeed", "nunique"),
        Retries=("ExtraIterations", lambda value: int(value.gt(0).sum())),
    ).reset_index()
    table["PostLDPC_BER"] = table.PayloadBitErrors / table.PayloadBits
    table["Goodput_Mbps"] = table.CorrectTransportBlocks * PAYLOAD_BITS / (
        table.Frames * SLOT_SECONDS * 1e6
    )
    return table


def paired_interval(rows, candidate, control):
    """Equal weight to each tested SNR. Resample complete channel realizations."""
    rng = np.random.default_rng(40583)
    group = rows.groupby(["ChannelSeed", "Estimator"])
    pairs = (group.Errors.sum() / group.PayloadBits.sum()).unstack()
    difference = (pairs[control] - pairs[candidate]).to_numpy()
    samples = rng.choice(difference, (10000, len(difference)), replace=True).mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def figures(table, methods, choice, output):
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    styles = {
        "LS": ("#a0522d", "s", "-"), "LMMSE": ("#377eb8", "^", "-"),
        "Matched LMMSE": ("#5c6d9b", "v", "--"), "V3 CNN": ("#4c9077", "d", "--"),
        "Perfect CSI": ("#353535", "x", ":"), choice: ("#d02e3e", "o", "-"),
    }
    for bs in [32, 64]:
        subset = table[table.BS == bs]
        for metric, tag, ylabel in [
            ("PostLDPC_BER", "ber", "Payload BER after LDPC"),
            ("Goodput_Mbps", "goodput", "Payload goodput (Mbit/s)"),
        ]:
            fig, ax = plt.subplots(figsize=(9.5, 6.0), layout="constrained")
            for method in methods:
                values = subset[subset.Estimator == method].sort_values("SNR_dB")
                color, mark, linestyle = styles[method]
                label = "CNN" if method == choice else method
                y = values[metric].to_numpy()
                if tag == "ber":
                    zero = y == 0
                    resolution = 1 / values.PayloadBits.to_numpy()
                    y = np.where(zero, resolution, y)
                    if zero.any():
                        ax.scatter(values.SNR_dB[zero], y[zero], marker="v", s=54,
                                   facecolors="none", edgecolors=color, zorder=6)
                ax.plot(values.SNR_dB, y, label=label, color=color, marker=mark,
                        linestyle=linestyle, linewidth=1.8, markersize=4)
            ax.set_xlim(0, 20)
            ax.set_xticks(np.arange(0, 21, 2))
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"PZF | {bs} BS antennas, 4 UEs × 4 Rx | TDL-C, 512-QAM, LDPC 1/8")
            if tag == "ber":
                ax.set_yscale("log")
                floor = 10 ** np.floor(np.log10(1 / subset.PayloadBits.max()))
                ax.set_ylim(floor, .75)
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.text(.02, .025, "▽  Zero observed errors shown at one-error resolution",
                        transform=ax.transAxes, color="#444444", fontsize=9)
            else:
                ax.set_ylim(0, 51)
            ax.grid(True, which="major", color="#cdd1d5", alpha=.8)
            ax.grid(True, which="minor", axis="y", color="#e6e8ea", alpha=.65)
            ax.legend(loc="upper right" if tag == "ber" else "upper left",
                      ncol=2, frameon=True, framealpha=.96)
            for ext in ["png", "pdf"]:
                fig.savefig(output / f"pzf_{bs}_{tag}.{ext}", dpi=210)
            plt.close(fig)


def report(root, config, output):
    choice = json.loads(Path(config).read_text())
    cnn = choice["cnn"]
    retry = choice["common_receiver"]["retry"]
    modes = list(dict.fromkeys(["None", retry]))
    methods = list(dict.fromkeys(["LS", "LMMSE", "Matched LMMSE", "V3 CNN", cnn, "Perfect CSI"]))
    rows = load_rows(root)
    check_design(rows, seeds(90000000, 8), TEST_SNR, methods, modes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "pzf_paired_slots.csv", index=False)
    table = aggregate(rows)
    table.to_csv(output / "pzf_metrics.csv", index=False)
    chosen = table[table.Retry == retry]
    final_rows = rows[rows.Retry == retry]
    curves = chosen.pivot(index=["BS", "SNR_dB"], columns="Estimator", values="PostLDPC_BER")
    comparison = {}
    for control in ["LS", "LMMSE", "Matched LMMSE", "V3 CNN"]:
        if control == cnn:
            continue
        comparison[control] = {
            "mean_BER_control": float(curves[control].mean()),
            "mean_BER_CNN": float(curves[cnn].mean()),
            "paired_win_points": int((curves[cnn] < curves[control]).sum()),
            "paired_tie_points": int((curves[cnn] == curves[control]).sum()),
            "point_count": len(curves),
            "mean_BER_reduction_CI95": paired_interval(final_rows, cnn, control),
        }
    sources = {}
    for path in sorted(Path("models").rglob("pzf_receiver.mat")):
        sources[path.parent.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "configuration": choice, "frames_per_array": 32, "SNR_dB": TEST_SNR,
        "one_error_resolution": 1 / (32 * 4 * PAYLOAD_BITS),
        "maximum_goodput_Mbps": 4 * PAYLOAD_BITS / (SLOT_SECONDS * 1e6),
        "comparisons": comparison, "receiver_model_sha256": sources,
        "undetected_blocks": int(final_rows.Undetected.sum()),
        "at_20_dB": chosen[chosen.SNR_dB == 20].to_dict("records"),
    }
    write_json(output / "pzf_review.json", summary)
    figures(chosen, methods, cnn, output)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, args in [
        ("receiver", ["root", "out"]),
        ("select", ["root", "receiver_choice", "out"]),
        ("replay", ["new_root", "old_root"]),
        ("report", ["root", "config", "output"]),
    ]:
        command = sub.add_parser(name)
        for arg in args:
            command.add_argument(arg)
    params = vars(parser.parse_args())
    stage = params.pop("stage")
    {"receiver": select_receiver, "select": select_cnn, "replay": replay, "report": report}[stage](**params)


if __name__ == "__main__":
    main()
