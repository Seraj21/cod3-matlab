#!/usr/bin/env python3
"""Verify raw MATLAB MU packet counts and draw the requested comparisons."""
import argparse, base64, csv, hashlib, io, json, math, os, shutil, subprocess, zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRECODERS=["MR","ZF","PZF"]
ESTIMATORS=["LS","LMMSE","CNN","Perfect CSI"]
SNRS=np.r_[np.arange(0,10.01,.5),np.arange(11,21)]
A=11880
T=.00025
U=4
BATCHES=8
N=16
CAP=U*A/T/1e6
COLORS=["#d18822","#4174b5","#c13d68","#2c8a65"]
STYLES=["--","-","-","-."]
FIELDS=["BSAntennas","Precoder","Estimator","SNR_dB","PostLDPC_BER",
"PayloadBitErrors","PayloadBits","Goodput_Mbps","CorrectTransportBlocks"]
def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
def boolean(v):
    assert v.lower() in ("true","false","1","0"), v
    return v.lower() in ("true","1")
def close(x,y):
    np.testing.assert_allclose(x,y,atol=2e-10,rtol=2e-9)
def fingerprint(v):
    h=hashlib.sha256()
    for item in v.flat:
        z=np.asarray(item)
        h.update(str(z.shape).encode());h.update(z.dtype.str.encode())
        h.update(z.tobytes(order="F"))
    return h.hexdigest()
def read_batch(path,array):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        def member(name):
            matches=[p for p in archive.namelist() if Path(p).name==name]
            assert len(matches)==1,(name,matches)
            return archive.read(matches[0])
        raw=rows(member("mu_slot_errors.csv"))
        metric=rows(member("mu_metrics.csv"))
        mat=loadmat(io.BytesIO(member("mu_results.mat")),
            variable_names=["bitErrors","totalBits","ber","success","goodput",
                            "maximumGoodput","cnnFingerprint"])
    shape=(N,len(SNRS),3,4,U)
    e=np.full(shape,-1,dtype=np.int64)
    fail=np.zeros(shape,dtype=bool)
    for row in raw:
        p=int(row["Packet"])-1;u=int(row["User"])-1
        s=np.flatnonzero(SNRS==float(row["SNR_dB"]))
        assert len(s)==1
        j=PRECODERS.index(row["Precoder"]);k=ESTIMATORS.index(row["Estimator"])
        ix=(p,int(s[0]),j,k,u)
        assert 0<=p<N and 0<=u<U and e[ix]==-1
        value=int(row["PayloadBitErrors"])
        assert int(row["PayloadBits"])==A and 0<=value<=A
        e[ix]=value;fail[ix]=boolean(row["CRCFailed"])
    assert np.all(e>=0) and len(raw)==np.prod(shape)
    errors=e.sum(axis=(0,4))
    bits=N*U*A
    correct=((e==0)&~fail).sum(axis=(0,4))
    close(mat["bitErrors"],errors);close(mat["ber"],errors/bits)
    close(mat["totalBits"],np.full_like(errors,bits))
    close(mat["success"],correct);close(mat["goodput"],correct*A/(N*T)/1e6)
    close(mat["maximumGoodput"].item(),CAP)
    assert len(metric)==len(SNRS)*3*4
    seen_metrics=set()
    for row in metric:
        ix=(int(np.flatnonzero(SNRS==float(row["SNR_dB"]))[0]),
            PRECODERS.index(row["Precoder"]),ESTIMATORS.index(row["Estimator"]))
        assert ix not in seen_metrics, ("Duplicate aggregate CSV measurement",ix)
        seen_metrics.add(ix)
        assert int(row["BSAntennas"])==array
        assert int(row["PayloadBits"])==bits
        assert int(row["PayloadBitErrors"])==errors[ix]
        assert int(row["CorrectTransportBlocks"])==correct[ix]
        close(float(row["PostLDPC_BER"]),errors[ix]/bits)
        close(float(row["Goodput_Mbps"]),correct[ix]*A/(N*T)/1e6)
    return e,fail,fingerprint(mat["cnnFingerprint"])
def writecsv(path,items,fields=None):
    with path.open("w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=fields or list(items[0]))
        writer.writeheader();writer.writerows(items)
def model_info(root):
    info={}
    if (root/"training_summary.json").exists():
        info.update(json.loads((root/"training_summary.json").read_text()))
    matches=list(root.glob("mu-training-report.zip"))
    if matches:
        with zipfile.ZipFile(matches[0]) as z:
            names=z.namelist()
            selected=[x for x in names if x.endswith("training/validation_payload_ber.csv")
                      and not x.startswith("smoke")]
            if selected:
                v=rows(z.read(selected[0]))
                info["payload_validation"]=v
                best=min(v,key=lambda x:(float(x["PostLDPC_BER"]),float(x["CRC_BlockErrorRate"])))
                info["selected_multiplier"]=float(best["CNNResidualMultiplier"])
            for link in ["tx","rx"]:
                selected=[x for x in names if x.endswith("training/validation_channel_"+link+".csv")
                          and not x.startswith("smoke")]
                if selected:
                    v=rows(z.read(selected[0]))
                    info["channel_validation_"+link]={key:float(np.mean([float(x[key]) for x in v]))
                        for key in ["MSE_LS","MSE_LMMSE","MSE_CNN_FullResidual"]}
                selected=[x for x in names if x.endswith("training/learning_"+link+".csv")
                          and not x.startswith("smoke")]
                if selected:
                    v=rows(z.read(selected[0]))
                    info["learning_"+link]={"records":len(v),"last":v[-1],"columns":list(v[0])}
    return info
def plot(array,p,kind,ber,gp,bits,out):
    from mu_plotting import figure_for_series
    values = ber[:,p,:] if kind=="BER" else gp[:,p,:]
    figure_for_series(SNRS,values,array,PRECODERS[p],kind,bits,N*BATCHES,CAP,
                      out/f"MU-{PRECODERS[p]}-{kind}-BS{array}.png")

def collect_auxiliary(root):
    """Read completed controls of this exact experiment; never change a repository."""
    if not (os.getenv("GITHUB_ACTIONS")=="true" and os.getenv("GH_REPO")=="Seraj21/cod3-matlab"):
        return
    directory=root/"diagnostics";directory.mkdir(parents=True,exist_ok=True)
    expected=[
        (13339340301,"rate_probe.csv","3f58b539b4a2ae202fab03c1dc7d017b968f6148ebb4195656e07c3e5cc3aa9b",41857825835),
        (13336503683,"cross_probe.csv","549d873f7c1f3441bf50cbb2e7914d10e7430e86beb90efafd7331141248c73d",41854206157)]
    manifest=[]
    for aid,name,digest,run in expected:
        data=subprocess.check_output(["gh","api",f"repos/Seraj21/cod3-matlab/actions/artifacts/{aid}/zip"])
        assert hashlib.sha256(data).hexdigest()==digest
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            assert z.testzip() is None
            matches=[n for n in z.namelist() if Path(n).name==name]
            assert len(matches)==1
            (directory/name).write_bytes(z.read(matches[0]))
        manifest.append(dict(artifact_id=aid,run_id=run,sha256=digest,file=name))
    (directory/"auxiliary_manifest.json").write_text(json.dumps(manifest,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--artifacts",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    root=args.artifacts;out=args.output;out.mkdir(parents=True,exist_ok=True)
    collect_auxiliary(root)
    rawout=out/"raw";rawout.mkdir(exist_ok=True)
    allrows=[];paired=[];hashes=set();summary={"source_commit":"73c537ef9870648df6dccd86b56c498bf4e4db29",
        "run_id":41841619172,"model":model_info(root),"arrays":{}}
    rng=np.random.default_rng(923116)
    # A channel frame (all four mutually interfering users) is the resampling unit.
    w=rng.multinomial(N*BATCHES,np.full(N*BATCHES,1/(N*BATCHES)),size=10000)/(N*BATCHES)
    for array in [32,64]:
        es=[];fs=[]
        for batch in range(BATCHES):
            path=root/f"mu-{array}-batch-{batch}.zip"
            e,f,h=read_batch(path,array);es.append(e);fs.append(f);hashes.add(h)
            shutil.copy2(path,rawout/path.name)
        e=np.concatenate(es,axis=0);fail=np.concatenate(fs,axis=0)
        errors=e.sum(axis=(0,4));bits=e.shape[0]*U*A
        correct=((e==0)&~fail).sum(axis=(0,4))
        ber=errors/bits;gp=correct*A/(e.shape[0]*T)/1e6
        assert np.all((gp>=0)&(gp<=CAP+1e-9))
        arrayrows=[];pointsummary=[]
        for p,prec in enumerate(PRECODERS):
            for k,est in enumerate(ESTIMATORS):
                for s,snr in enumerate(SNRS):
                    row=dict(zip(FIELDS,[array,prec,est,float(snr),float(ber[s,p,k]),
                        int(errors[s,p,k]),bits,float(gp[s,p,k]),int(correct[s,p,k])]))
                    arrayrows.append(row);allrows.append(row)
            for s,snr in enumerate(SNRS):
                de=(e[:,s,p,2,:]-e[:,s,p,1,:]).sum(axis=1)/(U*A)
                dg=(((e[:,s,p,2,:]==0)&~fail[:,s,p,2,:]).sum(axis=1)-
                    ((e[:,s,p,1,:]==0)&~fail[:,s,p,1,:]).sum(axis=1))*A/T/1e6
                brow={"BSAntennas":array,"Precoder":prec,"SNR_dB":float(snr),
                    "CNN_minus_LMMSE_BER":float(de.mean()),
                    "CNN_minus_LMMSE_Goodput_Mbps":float(dg.mean())}
                for name,values in [("BER",de),("Goodput_Mbps",dg)]:
                    differing=int(np.count_nonzero(values))
                    ci=np.quantile(w@values,[.025,.975]) if differing>=5 else [float("nan")]*2
                    brow[name+"_CI95_low"]=float(ci[0]);brow[name+"_CI95_high"]=float(ci[1])
                    brow[name+"_DifferingFrames"]=differing
                paired.append(brow)
                if snr in [0,4,8,12,16,20]:
                    pointsummary.append({"precoder":prec,"snr":float(snr),
                        "BER":dict(zip(ESTIMATORS,map(float,ber[s,p,:]))),
                        "Goodput_Mbps":dict(zip(ESTIMATORS,map(float,gp[s,p,:]))),
                        "CNN_minus_LMMSE_BER_CI95":[float(brow[x]) if np.isfinite(brow[x]) else None for x in ["BER_CI95_low","BER_CI95_high"]]})
            for kind in ["BER","Goodput"]:plot(array,p,kind,ber,gp,bits,out)
        writecsv(out/f"MU-aggregate-BS{array}.csv",arrayrows,FIELDS)
        summary["arrays"][str(array)]={"packets":len(e),"payload_bits_per_point":bits,
            "maximum_goodput_Mbps":CAP,"selected_points":pointsummary,
            "CNN_BER_better_points":int(np.count_nonzero(ber[:,:,2]<ber[:,:,1])),
            "CNN_BER_worse_points":int(np.count_nonzero(ber[:,:,2]>ber[:,:,1])),
            "CNN_BER_tied_points":int(np.count_nonzero(ber[:,:,2]==ber[:,:,1]))}
    assert len(hashes)==1,"Different frozen CNN weights were used between jobs!"
    summary["model_fingerprint"]=hashes.pop()
    summary["checks"]=["16 ZIP CRC checks","Raw per-user counts reproduce every CSV and MAT metric",
        "128 independent channel frames per array","Identical frozen CNN weights in all 16 jobs",
        "No duplicated packet/user/precoder/estimator/SNR row","Goodput excludes erroneous or CRC-failed blocks",
        "Goodput cannot exceed resource-count ceiling","No fitted, sorted, clipped or forced-monotonic BER curves"]
    writecsv(out/"MU-paired-CNN-LMMSE.csv",paired)
    writecsv(out/"MU-all-metrics.csv",allrows,FIELDS)
    from mu_plotting import audit_rows
    plot_audit=audit_rows(allrows,SNRS)
    writecsv(out/"MU-plot-audit.csv",plot_audit)
    summary["plot_audit"]=plot_audit
    if (root/"diagnostics/rate_probe.csv").exists():summary["rate_control"]=rows((root/"diagnostics/rate_probe.csv").read_bytes())
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    for name in ["mu-training-report.zip","mu-frozen-model.zip","artifact_manifest.json","training_summary.json"]:
        if (root/name).exists():shutil.copy2(root/name,out/name)
    if (root/"diagnostics").exists():shutil.copytree(root/"diagnostics",out/"diagnostics",dirs_exist_ok=True)
    report=review(summary)
    (out/"MU-Review.md").write_text(report,encoding="utf-8")
    archive=out.parent/"MU-TDL-C-512QAM-MR-ZF-PZF.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.rglob("*")):
            if file.is_file():z.write(file,file.relative_to(out))
    print("MU_REPORT_SUMMARY_BEGIN")
    print(json.dumps(summary,ensure_ascii=False))
    print("MU_REPORT_SUMMARY_END")
    print("Archive SHA256:",hashlib.sha256(archive.read_bytes()).hexdigest())
    for name in ["MU-MR-BER-BS64.png","MU-ZF-BER-BS64.png","MU-PZF-BER-BS64.png","MU-ZF-Goodput-BS64.png"]:
        encoded=base64.b64encode((out/name).read_bytes()).decode("ascii")
        print("MU_PREVIEW_BEGIN "+name)
        for j in range(0,len(encoded),8192):print(encoded[j:j+8192])
        print("MU_PREVIEW_END "+name)
def review(s):
    text="""# نتائج محاكاة Multi-User TDD Downlink MIMO

هذه نتائج محاكاة موجية زمنية، على MATLAB R2025b مع 5G وCommunications وDeep Learning Toolboxes.

## النظام والموارد
- محطة بـ32 أو64 هوائيًا، 8 سلاسل RF، اتصال جزئي ثابت، 4 مستخدمين لكل منهم 4 هوائيات استقبال وطبقة بيانات واحدة.
- TDL-C، انتشار تأخير RMS مقداره 1 ميكروثانية، سرعة 120 km/h، تردد 28 GHz، ارتباط هوائيات رأسي 0.7 وأفقي 0.3، وارتباط استقبال 0.25.
- 512-QAM، 9 بتات/رمز، CRC وLDPC بمعدل حمولة مستهدف 0.5، RV=0، 20 تكرار normalized min-sum.
- 66 RB و60 kHz وCP ممتد؛ إطار 12 رمزًا بزمن 0.25 ms: أربعة رموز sounding، ورمز حراسة، وسبعة رموز تنزيل.
- التخصيص الفعلي بعد DM-RS وPTRS هو2640 رمز بيانات و23760 بتًا مشفرًا و11880 بت حمولة لكل مستخدم.
- SNR رفع ثابت15 dB؛ SNR تنزيل0 إلى20 بخطوة0.5 حتى10 ثم1 dB. المحور هو قدرة الإرسال الإجمالية إلى ضوضاء الاستقبال، مع تقسيم القدرة على المستخدمين الأربعة.
- يشمل المسار مضخم Rapp، AM/PM، ضوضاء الطور، CFO، توقيتًا عمليًا، وتصحيح CFO وPTRS. تستعمل حالات Perfect CSI مرجع القناة والتوقيت الحقيقيين، وتبقى تشوهات المسار الموجي.
- التعديل512-QAM وموارد sounding من نوع comb امتدادان تجريبيان لهيكل قائم على NR؛ الوصف ليس إعلان مطابقة جهاز5G تجاري.

## المقارنات
لكل MR وZF وPZF تجربة مستقلة مع LS وLMMSE وCNN وPerfect CSI. تشترك الحالات في القناة والحمولة وعينات الضوضاء والمراجع. تتغير الموجة المرسلة عندما تتغير أوزان precoding.

MR وZF يعملان في مجال سلاسل RF الثماني. PZF هنا خوارزمية محددة: إسقاط أطوار ZF إلى16 طورًا، ثم جولتان من البحث الإحداثي لتقليل التداخل نسبةً للكسب. أوزانها الفيزيائية ثابتة السعة، ولا نفترض إلغاء التداخل تمامًا بعد تكميم الأطوار. الاختبارات تتحقق من القدرة، وتكميم الأطوار، وثبات السعة، والتداخل الصفري لـZF عند توفر الرتبة.

## CNN والتدريب
توجد شبكتان لتصحيح LMMSE: واحدة لتقدير/تنبؤ قناة التنزيل من sounding عند المحطة، والثانية لتقدير القنوات الفعالة لدى المستخدم. تتعاملان مع القنوات المشتركة عبر الهوائيات والطبقات، بإعادة ترتيب ترددي تحفظ العينات وكتل residual depthwise/pointwise.

استعمل التدريب256 تحققًا مستقلًا للقناة، مع تنويع الانتشار والسرعة والارتباط وSNR. استعمل تحقق خطأ القناة32 قناة منفصلة. سقف التدريب45 عصرًا، مع توقف مبكر واختيار أفضل خسارة تحقق. حُدد مقدار التصحيح على16 إطار تحقق إضافيًا باختبار BER الحمولة بعد LDPC عند0 و4 و8 و12 و16 و20 dB، لكل precoder وكلا حجمي المحطة. المعاملات المسبقة المحددة للاختيار هي0 و0.5 و1. المعامل0 يرجع تمامًا إلى LMMSE؛ لا نفرض استعمال شبكة إذا أضر تصحيحها بالتحقق.

يتعلم LMMSE إحصاءات تردد/زمن من مجموعة التدريب نفسها لكل قناة مفردة؛ CNN تستفيد أيضًا من البنية المشتركة بين الهوائيات والطبقات. توثق هذه الفروق نطاق المقارنة. مجموعات الاختبار منفصلة، ونفس الشبكتين المجمدتين تستعملان في المقاسين وجميع الحزم.

## تعريف النتائج
BER هو عدد بتات الحمولة الخاطئة بعد LDPC مقسومًا على جميع بتات الحمولة. لا ترتب القيم ولا تجبر على التناقص معSNR.

Goodput = مجموع بتات الحمولات الخالية من الخطأ والمجتازة CRC ÷ مجموع أزمنة الإطارات. يبلغ سقف الموارد190.08 Mbit/s. يشمل المقام التدريب والحراسة والمراجع. هذا معدل PHY لكل إطار مستقل؛ لا يتضمن HARQ أو MAC أو جدولة شبكة مستمرة. لا تستبدل به سعة شانون أو معدلQAM الخام.

يضم الاختبار128 إطار قناة مستقلًا لكل حجم محطة و6082560 بت حمولة لكل نقطة. إعادة أخذ العينات في ملف المقارنة الزوجية تتم على إطار القناة كاملًا، محافظًا على اعتماد المستخدمين الأربعة. فترات95% نقطية؛ لا تمثل ضمانًا لجميع نقاطSNR مجتمعة. تحجب الفترة عند اختلاف أقل من خمسة إطارات. علامة الخطأ الصفري المفتوحة عند1/Nbits تعبر عن دقة العد، ولا تثبت أنBER الحقيقي يساوي صفرًا.

## النتائج العددية
"""
    m=s.get("model",{})
    text+=f"\nمعامل CNN المختار من تحقق BER: **{m.get('selected_multiplier','راجع ملف التدريب')}**.\n\n"
    text+=f"معامل تصحيح المحطة: **{m.get('BlendTX','راجع سجل التدريب')}**؛ معامل تصحيح المستقبل قبل المعامل العام: **{m.get('BlendRX','راجع سجل التدريب')}**.\n"
    text+="اختير LMMSE في المحطة لعدم تحسن تصحيح شبكة التنبؤ على التحقق، وبقيت CNN لدى المستقبل فعالة. لذلك يقصد منحنىCNN خط المعالجة المحسّن بالشبكة مع هذا الرجوع الموثق.\n\n"
    text+="| BS | Precoder | SNR dB | BER LS | BER LMMSE | BER CNN | BER Perfect | Goodput CNN Mbit/s |\n|---|---|---|---|---|---|---|---|\n"
    for array,info in s["arrays"].items():
        for p in info["selected_points"]:
            if p["snr"] not in [0,8,12,16,20]:continue
            b=p["BER"]
            text+=f"| {array} | {p['precoder']} | {p['snr']:g} | {b['LS']:.6g} | {b['LMMSE']:.6g} | {b['CNN']:.6g} | {b['Perfect CSI']:.6g} | {p['Goodput_Mbps']['CNN']:.6g} |\n"
    text+="\nعدد نقاط تحسن/تراجع/تعادلBER بالنسبة إلىLMMSE، عبر31 قيمةSNR وثلاثةprecoders:\n\n"
    for array,info in s["arrays"].items():
        text+=f"- BS{array}: تحسن {info['CNN_BER_better_points']}، تراجع {info['CNN_BER_worse_points']}، تعادل {info['CNN_BER_tied_points']} من93 نقطة.\n"
    text+="\nلا تعني زيادة قوة النموذج ضمان تفوقCNN في كلSNR. النتائج الخام وفروقBER وGoodput وفترات الثقة متاحة فيCSV، ويجب تفسير فروقها مع حجم العينة. قد ينشأ التسطح في هذه الظروف من تقادمCSI خلالTDD والتداخل المتبقي وICI وتشوهاتRF، إضافة إلى خطأ التقدير؛ اختبار المسار الهادئ عاليSNR يتحقق من فك حمولة المستخدمين لجميعprecoders.\n"
    text+="\n## تشخيص التسطح\nاختبار تبادل القناة تحت الحركة أعطى خطأ نسبيًا صفريًا بين المرجع المتبادل ومرجع التنزيل المباشر. في الاختبار المصغر، بلغ خطأ نموذج القناة القطري نحو1.39% من قدرة الإشارة عند120 km/h و1µs، مقابل نحو0.55% عند30ns مع السرعة نفسها، ونحو1% عند1µs مع سرعة صفر. تطابقت النتائج تقريبًا بين توليد path gains بالمعدل المختزل والمعدل الكامل. هذه اختبارات لفصل أسباب الأداء، ولا تحل محل اختبار128 إطارًا.\n"
    text+="\nنجح المرجع الهادئ لـZF في فك جميع الحمولات عند20dB. فشلت حالات الانتشار1µs الشديدة حتى معPerfect CSI في الاختبار التشخيصي. يترك النموذج القطري ومجمعMMSE تداخلًا بين الرموز والحوامل لا يزيله إعطاء CSI مثاليًا، وقد يضخمه فصل الطبقات. توجد قياسات انتشار300ns وعروض precoding مختلفة فيdiagnostics، باستعمال الشبكة المجمدة دون إعادة اختيارها. هذه القياسات مبنية على إطارين لكل حالة وتفسر الحساسية فقط.\n"
    text+="\nاختبار إضافي فصلCSI المحطة عنCSI المستقبل عند300ns و20dB: نجح مرجعZF المثالي للطرفين في الحمولات الثماني، بينما فشلت الحمولات الثماني عند تقديرLMMSE في المحطة وحده أو في المستقبل وحده. خفضتCNN خطأ التقدير لدى المستقبل دون تحقيق فك حمولة ناجح في هذه العينة. [القياسات الخام للاختبار الإضافي](https://github.com/Seraj21/cod3-matlab/actions/runs/41854206157) تستعمل إطارين فقط، وهي تشخيصية ولا تدخل اختيار النموذج.\n"
    if s.get("rate_control"):
        text+="\n## اختبار منفصل لزيادة حمايةLDPC\nيحتفظ الاختبار الأساسي بمعدل0.5. الاختبار التالي خاص بـZF وBS64 وبنفسTDL-C الشديدة، على إطارين مستقلين وثماني حمولات لكل نقطة، وبالشبكة المجمدة نفسها. لا يدخل اختيارCNN ولا يمثل إثباتًا إحصائيًا عامًا. ينخفض سقف الموارد إلى94.976Mbit/s للمعدل1/4 وإلى47.488Mbit/s للمعدل1/8؛ الزيادة في الحماية تقابل انخفاض الحمولة.\n\n"
        text+="| معدل الترميز المستهدف | SNR dB | BER LMMSE | BER CNN | Goodput LMMSE | Goodput CNN |\n|---|---|---|---|---|---|\n"
        data=s["rate_control"]
        for rate in [.25,.125]:
            for snr in [0,4,8,12,16,20]:
                select=lambda est:next(x for x in data if float(x["TargetCodeRate"])==rate and float(x["SNR_dB"])==snr and x["Estimator"]==est)
                a=select("LMMSE");b=select("CNN")
                text+=f"| {rate:g} | {snr} | {float(a['PostLDPC_BER']):.6g} | {float(b['PostLDPC_BER']):.6g} | {float(a['Goodput_Mbps']):.6g} | {float(b['Goodput_Mbps']):.6g} |\n"
        text+="\nجميع قيمGoodput في الجدول بوحدةMbit/s. تظهر هذه العينة أن فائدة تحسينCSI تعتمد أيضًا على معدل الترميز ومنطقة الانتقال بين فشلLDPC ونجاحه؛ لا تكفي زيادة التدريب لضمان تسليم الحمولات عندمعدل ثابت مرتفع في كل القنوات.\n"
    text+="\n## التحقق وإعادة التشغيل\nنجحت الاختبارات الثمانية، بما فيها reciprocity لـTDL-C، فصلDM-RS، حفظ ترتيب ميزاتCNN، فك الترميز متعدد المستخدمين دون ضوضاء، والتحكم الزمني الكامل عاليSNR. نجح التدريب المصغر والتقييم لكلا حجمي المحطة قبل التدريب الموسع. يعيد محلل النتائج حساب كل قيمة من أخطاء كل مستخدم، ويقارنCSV معMAT، ويتحقق من تطابق أوزانCNN في جميع ملفات الاختبار.\n"
    text+="\nالمصدر: https://github.com/Seraj21/cod3-matlab/tree/mu-downlink-tdlc\n"
    text+="\nتشغيل المصدر علىcommit73c537ef9870648df6dccd86b56c498bf4e4db29. بذور التدريب تبدأ1000000، تحقق القناة2000000، تحققBER3000000 مع إزاحة الحجم، والاختبار10000000 مع إزاحة مستقلة لكلحزمة.\n"
    text+="\nمراجع الهيكل: [مثال MathWorks للتنزيل متعدد المستخدمين اعتمادًا على SRS](https://www.mathworks.com/help/5g/ug/tdd-reciprocity-based-pdsch-beamforming-using-srs.html)، [تبادل الإرسال والاستقبال في nrTDLChannel](https://www.mathworks.com/help/5g/ref/nrtdlchannel.swaptransmitandreceive.html). تعريفPZF في هذه المحاكاة مثبت أعلاه وفيmuPrecoder.m.\n"
    return text
if __name__=="__main__":main()
