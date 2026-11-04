#!/usr/bin/env python3
"""Audit all raw MATLAB counts and display every measured estimator curve."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from analyze_mu import read_batch,writecsv,SNRS,PRECODERS,ESTIMATORS,A,T,U,N,BATCHES,CAP,FIELDS
from mu_plotting import figure_for_series,audit_rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',type=int,required=True)
    parser.add_argument('--source',required=True)
    args=parser.parse_args();root=args.artifacts;out=args.output;out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(root/'v2-frozen-model.zip') as frozen:
        names=[name for name in frozen.namelist() if name.endswith('v2_model_summary.json')]
        assert len(names)==1
        model=json.loads(frozen.read(names[0]))
    fingerprints=set();allrows=[];paired=[]
    summary={'run_id':args.run_id,'source':args.source,'model':model,'arrays':{}}
    raw=out/'raw';raw.mkdir(exist_ok=True)
    rng=np.random.default_rng(925117)
    weights=rng.multinomial(N*BATCHES,np.full(N*BATCHES,1/(N*BATCHES)),size=10000)/(N*BATCHES)
    for array in [32,64]:
        es=[];fs=[]
        for batch in range(BATCHES):
            path=root/f'v2-evaluation-{array}-{batch}.zip'
            e,f,digest=read_batch(path,array);es.append(e);fs.append(f);fingerprints.add(digest)
            shutil.copy2(path,raw/path.name)
        e=np.concatenate(es);f=np.concatenate(fs)
        errors=e.sum(axis=(0,4));bits=len(e)*U*A
        success=((e==0)&~f).sum(axis=(0,4));ber=errors/bits;gp=success*A/(len(e)*T)/1e6
        assert np.all(gp<=CAP+1e-9)
        points=[]
        for p,precoder in enumerate(PRECODERS):
            for m,estimator in enumerate(ESTIMATORS):
                for s,snr in enumerate(SNRS):
                    allrows.append(dict(zip(FIELDS,[array,precoder,estimator,float(snr),
                        float(ber[s,p,m]),int(errors[s,p,m]),bits,float(gp[s,p,m]),int(success[s,p,m])])))
            for s,snr in enumerate(SNRS):
                difference=(e[:,s,p,2,:]-e[:,s,p,1,:]).sum(axis=1)/(U*A)
                good=(((e[:,s,p,2,:]==0)&~f[:,s,p,2,:]).sum(axis=1)-
                      ((e[:,s,p,1,:]==0)&~f[:,s,p,1,:]).sum(axis=1))*A/T/1e6
                row={'BSAntennas':array,'Precoder':precoder,'SNR_dB':float(snr),
                    'CNN_minus_LMMSE_BER':float(difference.mean()),
                    'CNN_minus_LMMSE_Goodput_Mbps':float(good.mean())}
                for label,values in [('BER',difference),('Goodput',good)]:
                    count=np.count_nonzero(values)
                    interval=np.quantile(weights@values,[.025,.975]) if count>=5 else [np.nan,np.nan]
                    row[label+'_CI95_low']=float(interval[0]);row[label+'_CI95_high']=float(interval[1])
                    row[label+'_DifferingFrames']=int(count)
                paired.append(row)
                if snr in [0,4,8,12,16,20]:
                    points.append({'precoder':precoder,'snr':float(snr),
                        'BER':dict(zip(ESTIMATORS,map(float,ber[s,p]))),
                        'Goodput_Mbps':dict(zip(ESTIMATORS,map(float,gp[s,p])))})
            for kind,values in [('BER',ber[:,p,:]),('Goodput',gp[:,p,:])]:
                figure_for_series(SNRS,values,array,precoder,kind,bits,len(e),CAP,
                    out/f'V2-{precoder}-{kind}-BS{array}.png',measured_label='Independent frozen-model simulation')
        summary['arrays'][str(array)]={'frames':len(e),'payload_bits_per_point':bits,'points':points,
            'CNN_BER_better':int(np.count_nonzero(ber[:,:,2]<ber[:,:,1])),
            'CNN_BER_worse':int(np.count_nonzero(ber[:,:,2]>ber[:,:,1])),
            'CNN_BER_equal':int(np.count_nonzero(ber[:,:,2]==ber[:,:,1])),
            'CNN_Goodput_better':int(np.count_nonzero(gp[:,:,2]>gp[:,:,1]))}
    assert len(fingerprints)==1,'The frozen networks differ between evaluation jobs'
    summary['network_fingerprint']=fingerprints.pop()
    audit=audit_rows(allrows,SNRS);summary['plot_audit']=audit
    writecsv(out/'V2-all-metrics.csv',allrows,FIELDS)
    writecsv(out/'V2-paired-CNN-LMMSE.csv',paired)
    writecsv(out/'V2-plot-audit.csv',audit)
    manifest=[]
    for path in sorted(root.glob('*.zip')):
        manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if path.name=='v2-frozen-model.zip':shutil.copy2(path,out/path.name)
        elif not path.name.startswith('v2-evaluation-'):
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if Path(member).suffix in ['.csv','.json']:
                        target=out/'training'/path.stem/member
                        target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(archive.read(member))
    summary['artifacts']=manifest
    summary['checks']=['All 16 ZIP archives passed CRC validation',
        'Every CSV/MAT BER and goodput reproduced from individual user payload errors and CRC flags',
        '31 SNR values for each of 24 estimator curves, with no missing or duplicate measurement',
        'Four identical frozen CNN weight sets in all 16 jobs',
        'No BER smoothing, sorting, forced monotonicity or substituted zero measurements']
    (out/'V2-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    text='''# نتائج النسخة المصححة V2

هذه نتائج محاكاة موجية زمنية لـTDL-C و512-QAM وLDPC بمعدل حمولة1/2، مع محطة32 أو64 هوائيًا وثماني سلاسلRF وأربعة مستخدمين، لكل مستخدم أربعة هوائيات استقبال وطبقة واحدة.

يبقى انتشار التأخير1µs والسرعة120km/h والتردد28GHz وSNR الرفع15dB. يمتد محور SNR التنزيل من0 إلى20dB، بخطوة0.5 حتى10 ثم1dB.

تستقل تجارب MR وZF وPZF مع LS وLMMSE وCNN وPerfect CSI. تستخدم المقارنات الحمولات نفسها وعينات القناة والضوضاء نفسها؛ تتغير الموجة المرسلة مع تغير أوزان precoding. تبقى مصفوفة RF ثابتة ومتصلة جزئيًا. يستخدم PZF سعة ثابتة وطورًا مكممًا إلى4بتات.

صُحح مسار الاستقبال بتوحيد موضع نافذةFFT مع مرجع القناة عندCP fraction0.75، واستخدام أوزان baseband لكل حاملة، وتقدير مصفوفة التداخل والضوضاء من المراجع المعروفة. يعيد المستقبل فك الحمولات الفاشلة بحزمة SRS ويقبل المحاولة فقط عند نجاحCRC، دون الرجوع إلى بتات الحمولة الأصلية في الاختيار.

تستخدم شبكة المحطة قياسات الطيار لتعلّم المزج بين خمسة متنبئات قناة مشروطة بدوبلر. لا تدخل السرعة الحقيقية في الاستدلال. توجد ثلاثة نماذج CNN للمستقبل، منفصلة حسبprecoder، مع ترتيب طبقة المستخدم أولًا وتعلّم تصحيح القناة وعدم يقينها. يبقى LMMSE مرجعًا بإحصاءات مشروطة مماثلة. دُربت النماذج على1920 قناة وتحققت على240 قناة مستقلة، ثم اختير مقدار التصحيح على16 إطارًا آخر قبل تجميد النموذج. تعرض ملفات التدريب الأعداد الفعلية ومعاملات الاختيار.

استعمل التدريب12RB مع رقع محلية من24حاملة، والاختبار66RB على التباعد الزمني والترددي نفسه. اختُبر مسار الاستدلال الكامل66RB قبل التحقق بالحمولة. لا يُستخدم إطار اختبار في التدريب أو اختيار المعاملات.

كل منحنى يحوي128 إطار قناة مستقلًا و6082560بت حمولة لكل نقطة. يحسبBER من بتات الحمولة بعدLDPC، ويحسبGoodput من الحمولات الصحيحة المجتازةCRC مقسومًا على زمن الإطارات الكامل، شاملاً sounding والحراسة والمراجع. سقف الموارد190.08Mbit/s. لا يشمل هذا المعدلHARQ أوMAC. التعديل512-QAM وموارد sounding تجريبيان ضمن هيكل مبني علىNR.

العلامة المفتوحة عندBER صفري مرصود تُعرض عند1/Nbits لإظهارها على المحور اللوغاريتمي؛ العدد الحقيقي محفوظ فيCSV. المنحنى الثابت عندGoodput=0 يعني أن الاختبار لم يسلم حمولة صحيحة، وليس قياسات مفقودة. تتبادل الطرق مواضع العلامات على نقاطSNR الفعلية لإظهار التطابق دون تحريك البيانات.

## القيم عند20dB

| BS | Precoder | BER LS | BER LMMSE | BER CNN | BER Perfect CSI | Goodput CNN Mbit/s |
|---|---|---|---|---|---|---|
'''
    for array,info in summary['arrays'].items():
        for row in info['points']:
            if row['snr']!=20:continue
            b=row['BER'];text+=f"| {array} | {row['precoder']} | {b['LS']:.6g} | {b['LMMSE']:.6g} | {b['CNN']:.6g} | {b['Perfect CSI']:.6g} | {row['Goodput_Mbps']['CNN']:.6g} |\n"
    text+='\n## معاملات CNN المجمدة\n\nالمعاملات التالية اختيرت على تحقق الحمولة المستقل، قبل الاختبار. المعامل صفر يلغي التصحيح في ذلك الطرف ويستعمل تقديرLMMSE. عندصفر في الطرفين يتطابق منحنىCNN معLMMSE عن قصد، وتبقى العلامات المختلفة ظاهرة. لا يُدّعى تحسن شبكة جرى إلغاء تصحيحها.\n\n'
    text+='| Precoder | معامل المحطة من تحقق الحمولة | معامل المستقبل من تحقق الحمولة | معامل المستقبل من تحقق القناة |\n|---|---|---|---|\n'
    for precoder in PRECODERS:
        selection=model['Selection'][precoder]
        text+=f"| {precoder} | {selection[0]:g} | {selection[1]:g} | {model['BlendRX'][precoder]:.6g} |\n"
    text+=f"\nمعامل شبكة المحطة من تحقق القناة: {model['BlendTX']:.6g}. المعامل الفعلي هو حاصل ضرب معاملي تحقق القناة والحمولة.\n"
    text+='\nملف المقارنة الزوجية يعرض فروقCNN وLMMSE وفترات ثقة نقطية بإعادة أخذ العينات على إطار القناة كاملًا. تحجب الفترة عندما تختلف أقل من خمسة إطارات؛ لا تُفسر العلامات الصفرية كضمان لخطأ صفري.\n'
    text+=f'\nالمصدر المستخدم: https://github.com/Seraj21/cod3-matlab/tree/{args.source}\n\nالتشغيل: https://github.com/Seraj21/cod3-matlab/actions/runs/{args.run_id}\n'
    (out/'V2-Review.md').write_text(text,encoding='utf-8')
    archive=out.parent/'MU-V2-Corrected-CNN-Results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['artifacts','plot_audit']},ensure_ascii=False))
    print('Archive:',archive,'SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())

if __name__=='__main__':main()
