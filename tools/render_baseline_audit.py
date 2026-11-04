"""Re-render the previous measured baseline without changing its values."""
import csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np
from mu_plotting import figure_for_series, audit_rows

root=Path(__file__).resolve().parents[1]/"plot-audit"
source=root/"MU-all-metrics.csv"
rows=list(csv.DictReader(source.open(encoding="utf-8-sig")))
snr=np.r_[np.arange(0,10.01,.5),np.arange(11,21)]
audit=audit_rows(rows,snr)
with (root/"MU-plot-audit.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
for array in [32,64]:
    for precoder in ["MR","ZF","PZF"]:
        selected={(r["Estimator"],float(r["SNR_dB"])):r for r in rows
                  if int(r["BSAntennas"])==array and r["Precoder"]==precoder}
        names=["LS","LMMSE","CNN","Perfect CSI"]
        for kind,field in [("BER","PostLDPC_BER"),("Goodput","Goodput_Mbps")]:
            values=np.array([[float(selected[name,float(x)][field]) for name in names] for x in snr])
            bits=int(selected["LS",0.0]["PayloadBits"])
            figure_for_series(snr,values,array,precoder,kind,bits,128,190.08,
                root/f"MU-{precoder}-{kind}-BS{array}.png",
                "PREVIOUS BASELINE: corrected display; original measurements retained")
description="""# تدقيق عرض القياسات السابقة

هذا الملف يصحح عرض النتائج السابقة. تدريب النسخة الجديدة وتحسين مستقبلها عمل منفصل.

فُحصت744 قيمة قياس: مجموعتا هوائيات، ثلاثةprecoders، أربعةمقدّرات، و31 قيمةSNR.
لكل مقياس24 سلسلة بيانات؛ جميعها كاملة، ولم تُستبدل قيمة مفقودة بصفر.
كل منحنياتBER السابقة متغيرة. جميع منحنياتGoodput السابقة تساوي صفرًا لأن عدد الحمولات الصحيحة المجتازةCRC يساوي صفرًا.

تعرض الرسوم الجديدة علامات الطرق الأربع في مواقعSNR متبادلة، مع إبقاء كل نقطة عند قيمتها الأصلية. توجد مساحة تحت خطGoodput=0 لإظهار العلامات كاملة. مقياسBER لوغاريتمي وحدوده تتبع نطاق البيانات. مربع وصفالطرق داخل الرسم؛ لا توجد مربعات تكبير أو أرقام إضافية.

الملفMU-all-metrics.csv هو البيانات الأصلية نفسها. MU-plot-audit.csv يسجل اكتمال كل سلسلة وثباتها. تصحيح العرض لا يثبت تحسن أداءالنظام؛ يقاس ذلك بنتائج النسخة الجديدة المستقلة.
"""
(root/"README.md").write_text(description,encoding="utf-8")
(root/"data-integrity.json").write_text(json.dumps(dict(
    OriginalCSV_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
    MetricRows=len(rows),CurvesPerMetric=len(audit),PointsPerCurve=31,
    MissingCurves=0,ConstantBER=sum(x["ConstantBER"] for x in audit),
    ZeroGoodputCurves=sum(x["AllGoodputZero"] for x in audit)),indent=2))
archive=root.parent/"MU-Previous-Baseline-Plot-Correction.zip"
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.iterdir()):
        if p.is_file():z.write(p,p.name)
print((root/"data-integrity.json").read_text())
print(archive)
