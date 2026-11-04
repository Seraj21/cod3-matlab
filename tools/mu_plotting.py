"""Render measured curves, keeping coincident or constant series visible."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

NAMES = ["LS", "LMMSE", "CNN", "Perfect CSI"]
COLORS = ["#be7615", "#356db3", "#c63267", "#238160"]
STYLES = ["--", "-", ":", "-."]
MARKERS = ["s", "o", "^", "D"]

def figure_for_series(x, values, array, precoder, kind, payload_bits, frames,
                      ceiling, destination, measured_label=""):
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    assert values.shape == (len(x), 4)
    assert len(x) >= 2 and np.all(np.diff(x) > 0)
    assert np.all(np.isfinite(values)), "A missing measurement must not be drawn as zero"
    assert np.all(values >= 0)
    fig, ax = plt.subplots(figsize=(10.4, 6.5), layout="constrained")
    shown = values.copy()
    if kind == "BER":
        assert np.all(values <= 1)
        shown[shown == 0] = 1 / payload_bits
    for method, name in enumerate(NAMES):
        ax.plot(x, shown[:, method], label=name, color=COLORS[method],
                linestyle=STYLES[method], linewidth=2, zorder=2+method)
        # Disjoint marker locations expose all four series even when lines coincide.
        locations = np.arange(method, len(x), 4)
        ax.scatter(x[locations], shown[locations, method], s=45,
                   marker=MARKERS[method], facecolors="white",
                   edgecolors=COLORS[method], linewidths=1.6,
                   zorder=10+method, clip_on=False)
        zero = (values[:, method] == 0) if kind == "BER" else np.zeros(len(x), bool)
        if np.any(zero):
            ax.scatter(x[zero], shown[zero, method], marker=MARKERS[method],
                       facecolors="none", edgecolors=COLORS[method], s=65, zorder=9)
    if kind == "BER":
        lower = max(1e-12, float(shown.min())*.85)
        upper = min(1, float(shown.max())*1.18)
        if upper <= lower: upper = min(1, lower*2)
        ax.set_yscale("log"); ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 2.5, 3, 4, 5, 6, 8]))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
        ax.set_ylabel("Payload BER after LDPC (logarithmic scale)")
        foot = f"{payload_bits:,} payload bits/point; {frames} channel frames. Open zero-error marks show 1/Nbits."
    else:
        peak = float(values.max())
        upper = max(1, peak*1.12)
        ax.set_ylim(-.045*upper, upper)
        ax.set_yticks(np.linspace(0, upper, 6))
        ax.set_ylabel("Sum payload goodput (Mbit/s)")
        ax.axhline(0, color="#777777", linewidth=.7, zorder=1)
        if peak == 0:
            ax.text(.5, .48, "All four methods: 0 Mbit/s\nNo correct CRC-passed payload blocks",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#555555")
        foot = f"Full frame includes sounding, guard and pilots. Resource ceiling: {ceiling:g} Mbit/s."
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_xlabel("Downlink SNR (dB)")
    ax.set_title(f"{precoder} | BS {array}, 4 users, 4 Rx/user | TDL-C, 512-QAM, LDPC")
    ax.grid(True, which="both", alpha=.22)
    ax.legend(loc="best", fontsize=11, framealpha=.95)
    fig.supxlabel((measured_label+"\n" if measured_label else "")+foot,
                  fontsize=8, color="#444444")
    fig.savefig(destination, dpi=170)
    plt.close(fig)

def audit_rows(rows, expected_snr):
    """Count completeness and distinguish zero throughput from absent data."""
    expected_snr = np.asarray(expected_snr, dtype=float)
    groups = {}
    for row in rows:
        key = (int(row["BSAntennas"]), row["Precoder"], row["Estimator"])
        groups.setdefault(key, []).append(row)
    assert len(groups) == 24, f"Expected 24 parameter/estimator curves, found {len(groups)}"
    result = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda r:float(r["SNR_dB"]))
        x = np.array([float(r["SNR_dB"]) for r in group])
        assert np.array_equal(x, expected_snr), f"Missing or duplicate SNR measurements: {key}"
        ber = np.array([float(r["PostLDPC_BER"]) for r in group])
        gp = np.array([float(r["Goodput_Mbps"]) for r in group])
        assert np.isfinite(ber).all() and np.isfinite(gp).all()
        for row, b in zip(group, ber):
            count, total = int(row["PayloadBitErrors"]), int(row["PayloadBits"])
            assert 0 <= count <= total and abs(b-count/total) < 1e-12
            assert float(row["Goodput_Mbps"]) != 0 or int(row["CorrectTransportBlocks"]) == 0
        result.append(dict(BS=key[0], Precoder=key[1], Estimator=key[2],
                           Points=len(group), MissingPoints=0,
                           ConstantBER=bool(np.ptp(ber)==0),
                           ConstantGoodput=bool(np.ptp(gp)==0),
                           AllGoodputZero=bool(np.all(gp==0))))
    return result
