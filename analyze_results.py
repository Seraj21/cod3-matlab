import csv
import json
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
SNR = np.r_[np.arange(0,10.01,.5), np.arange(11,21)]
NAMES = ("LS","LMMSE","CNN","Perfect CSI + timing")
STYLES = (("o","#2363a2","-"),("s","#24734d","--"),("D","#ab4318","-"),("^","#555261",":"))
FIELDS = ["M","arraySize","numSlots","snrDb","names","slotErrors","postBitErrors",
          "postBer","totalPayloadBits","payloadBitsPerSlot","cnnFingerprint","testSeedOffset",
          "timingAbsError","cfoResidual","config"]


def table(payload):
    return list(csv.DictReader(StringIO(payload.decode("utf-8"))))


def text_table(headers,rows):
    stream=StringIO(newline="")
    writer=csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def read_result(path,size):
    with ZipFile(path) as z:
        assert z.testzip() is None
        payload={name:z.read(name) for name in z.namelist()}
    required={"post_ldpc_ber.mat","post_ldpc_ber.csv","slot_errors.csv","snr_calibration.csv"}
    assert required.issubset(payload)
    d=loadmat(BytesIO(payload["post_ldpc_ber.mat"]),simplify_cells=True,variable_names=FIELDS)
    assert d["M"]==512 and d["arraySize"]==size and tuple(d["names"])==NAMES
    assert d["config"]["Version"]==3, "Stale receiver-frame model"
    slots=int(d["numSlots"])
    points=np.atleast_1d(d["snrDb"]).astype(float)
    bits=int(d["payloadBitsPerSlot"])
    counts=np.asarray(d["slotErrors"],dtype=np.int64).reshape(slots,len(points),4)
    assert np.all((counts>=0)&(counts<=bits))
    total=counts.sum(axis=0).T
    assert np.array_equal(total,np.asarray(d["postBitErrors"]).reshape(4,len(points)))
    assert np.allclose(total/(slots*bits),np.asarray(d["postBer"]).reshape(4,len(points)),atol=1e-16)
    for row in table(payload["post_ldpc_ber.csv"]):
        i=NAMES.index(row["Estimator"])
        j=int(np.flatnonzero(points==float(row["SNR_dB"])).item())
        assert int(row["PostLDPC_BitErrors"])==total[i,j]
        assert int(row["TotalPayloadBits"])==slots*bits
    rows=table(payload["slot_errors.csv"])
    assert len(rows)==counts.size
    seen=set()
    for row in rows:
        key=(int(row["Slot"])-1,int(np.flatnonzero(points==float(row["SNR_dB"])).item()),
             NAMES.index(row["Estimator"]))
        assert key not in seen
        seen.add(key)
        assert int(row["PostLDPC_BitErrors"])==counts[key] and int(row["PayloadBits"])==bits
    calibration=table(payload["snr_calibration.csv"])
    assert len(calibration)==slots
    for row in calibration:
        assert abs(10*np.log10(float(row["RFNoisePowerAt0dB"])))<.15
        assert 0<float(row["PAOutputPowerRatio"])<=1.000001
        assert int(row["SelectedBank"]) in (1,2)
    return dict(raw=payload,points=points,bits=bits,counts=counts,seed=int(d["testSeedOffset"]),
                fingerprint=d["cnnFingerprint"],calibration=calibration,
                cfo=np.asarray(d["cfoResidual"]),frequency_offset=d["config"]["Settings"]["FrequencyOffset"])


def aggregate(case,size):
    prefix="practical-hbf" if case=="CDL-C" else "deepmimo-ber"
    paths=sorted(ROOT.glob(f"{prefix}-{size}-batch-*.zip"))
    expected=6 if case=="CDL-C" else 3
    assert len(paths)==expected,(case,size,len(paths))
    refinement=sorted(ROOT.glob(f"refine-{prefix}-{size}-batch-*.zip"))
    assert len(refinement)==(6 if case=="CDL-C" else 0)
    paths+=refinement
    runs=[read_result(path,size) for path in paths]
    bits=runs[0]["bits"]
    seeds=[r["seed"] for r in runs]
    assert len(set(seeds))==len(seeds)
    for run in runs:
        assert run["bits"]==bits
        assert len(run["fingerprint"])==len(runs[0]["fingerprint"])
        assert all(np.array_equal(x,y) for x,y in zip(run["fingerprint"],runs[0]["fingerprint"]))
    errors=np.zeros((4,len(SNR)),dtype=np.int64)
    payload_bits=np.zeros(len(SNR),dtype=np.int64)
    rows=[]
    rng=np.random.default_rng(5220+size+(case=="DeepMIMO"))
    for j,s in enumerate(SNR):
        samples=[r["counts"][:,int(np.flatnonzero(r["points"]==s).item()),:]
                 for r in runs if s in r["points"]]
        counts=np.concatenate(samples)
        assert len(counts)==(192 if case=="CDL-C" and 3<=s<=12 else 96)
        errors[:,j]=counts.sum(axis=0)
        payload_bits[j]=len(counts)*bits
        delta=(counts[:,1]-counts[:,2])/bits
        weights=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),size=10000)
        ci=np.quantile(weights@delta/len(counts),[.025,.975])
        if np.count_nonzero(delta)<5:
            ci=[None,None]
        rows.append([case,size,s,len(counts),int(np.sum(delta>0)),int(np.sum(delta==0)),
                     int(np.sum(delta<0)),float(delta.mean()),*ci])
    return runs,errors,errors/payload_bits,payload_bits,rows


def plot(case,size,ber,payload_bits):
    fig,ax=plt.subplots(figsize=(10,6.2),dpi=170,layout="constrained")
    lower=.5/payload_bits.max() if np.any(ber==0) else 10.0**np.floor(np.log10(ber.min()))
    upper=min(.6,10.0**np.ceil(np.log10(max(1e-6,ber.max()*1.1))))
    for label,y,(marker,color,line) in zip(NAMES,ber,STYLES):
        ax.semilogy(SNR,np.where(y>0,y,np.nan),marker=marker,color=color,linestyle=line,
                    linewidth=1.9,markersize=5,label=label)
        zero=y==0
        ax.scatter(SNR[zero],1/payload_bits[zero],marker=marker,edgecolors=color,facecolors="none",
                   s={"LS":55,"LMMSE":95,"CNN":40,"Perfect CSI + timing":150}[label])
    ax.set(xlim=(-.2,20.2),xticks=np.arange(0,21,2),
           xlabel="Nominal reference SNR (dB)",ylabel="Payload BER after LDPC",
           title=f"{size}×{size} | 512-QAM | {case} | practical hybrid beamforming")
    low=int(np.floor(np.log10(lower)))
    ax.set_yticks(10.0**np.arange(low,0))
    ax.set_ylim(lower,upper)
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(True,color="#dbe1e7")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right",fontsize=10,framealpha=.97)
    caption=("Open markers at 1/N indicate zero observed errors." if np.any(ber==0)
             else "96 held-out ray-traced positions | 120 km/h stress test")
    ax.text(.5,-.18,caption,
            transform=ax.transAxes,ha="center",fontsize=9)
    png=ROOT/f"Practical-HBF-BER-{case}-{size}x{size}.png"
    fig.savefig(png,facecolor="white")
    plt.close(fig)
    return png


def main():
    all_rows,paired,calibration,replays=[],[],[],[]
    frozen={}
    bundle=ROOT/"Practical-HBF-512QAM-BER.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        for case in ("CDL-C","DeepMIMO"):
            for size in (32,64):
                runs,errors,ber,bits,rows=aggregate(case,size)
                if case=="DeepMIMO":
                    with ZipFile(ROOT/f"rf-diagnostic-{size}.zip") as diagnostic:
                        raw=diagnostic.read("rf_ablation.csv")
                        ablation=table(raw)
                        assert len(ablation)==12
                        for row in ablation:
                            assert int(row["Array"])==size
                            assert int(row["RayIndex"]) in (1,85,94)
                            assert int(row["Condition"]) in (1,2,3,4)
                            if int(row["Condition"])==4:
                                ray=int(row["RayIndex"])-1
                                run=runs[ray//32]
                                index=int(np.flatnonzero(run["points"]==20).item())
                                batch_cfo=run["frequency_offset"]-run["cfo"][ray%32,index]
                                diagnostic_cfo=float(row["EstimatedCFO"])
                                assert abs(batch_cfo-diagnostic_cfo)<1e-3
                                batch_errors=int(run["counts"][ray%32,index,3])
                                diagnostic_errors=int(row["PayloadBitErrors"])
                                replays.append([size,ray+1,diagnostic_errors,batch_errors,
                                                batch_errors-diagnostic_errors,diagnostic_cfo,batch_cfo])
                        z.writestr(f"Diagnostics/{size}x{size}/rf_ablation.csv",raw)
                if size in frozen:
                    assert all(np.array_equal(x,y) for x,y in zip(frozen[size],runs[0]["fingerprint"]))
                else:
                    frozen[size]=runs[0]["fingerprint"]
                paired.extend(rows)
                print(case,size,"bits per point",bits.min(),bits.max())
                for j,s in enumerate(SNR):
                    print(s,"errors",errors[:,j],"BER",ber[:,j])
                    for i,method in enumerate(NAMES):
                        all_rows.append([case,size,s,method,ber[i,j],errors[i,j],bits[j]])
                for batch,run in enumerate(runs):
                    for name,raw in run["raw"].items():
                        z.writestr(f"{case}/{size}x{size}/batch-{batch}/{name}",raw)
                    for row in run["calibration"]:
                        calibration.append([case,size,batch,*row.values()])
                z.write(plot(case,size,ber,bits),f"Practical-HBF-BER-{case}-{size}x{size}.png")
        z.writestr("Post-LDPC-BER.csv",text_table(
            ["Scenario","Array","SNR_dB","Estimator","PostLDPC_BER","PayloadBitErrors","PayloadBits"],all_rows))
        z.writestr("Paired-CNN-LMMSE.csv",text_table(
            ["Scenario","Array","SNR_dB","Slots","CNNWins","Ties","CNNLosses","BER_LMMSE_minus_CNN",
             "Pointwise95CI_Lower","Pointwise95CI_Upper"],paired))
        z.writestr("Calibration.csv",text_table(
            ["Scenario","Array","Batch","Slot","TxMeanGridPower","PAOutputPowerRatio",
             "AntennaSignalPowerPerRE","RFSignalPowerPerRE","RFNoisePowerAt0dB","SelectedBank"],calibration))
        z.writestr("Diagnostics/replay_comparison.csv",text_table(
            ["Array","RayIndex","DiagnosticBitErrors","BatchBitErrors","Difference",
             "DiagnosticCFO_Hz","BatchCFO_Hz"],replays))
        z.write(ROOT/"Review-and-assumptions.md","Review-and-assumptions.md")
        z.write(ROOT/"state.json","Runs.json")
        z.write(ROOT/"artifact_manifest.json","Artifact-manifest.json")
        for asset in ["deepmimo-source.zip","frozen-receiver-32.zip","frozen-receiver-64.zip"]:
            if (ROOT/asset).exists():
                z.write(ROOT/asset,asset)
    with ZipFile(bundle) as z:
        assert z.testzip() is None
    print("Validated",bundle)


if __name__=="__main__":
    main()
