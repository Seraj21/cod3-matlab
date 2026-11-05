#!/usr/bin/env python3
"""Audit all raw MATLAB counts and display every measured estimator curve."""
import argparse,csv,hashlib,json,shutil,zipfile
import io
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import analyze_mu as raw_audit
from analyze_mu import read_batch,writecsv,SNRS,PRECODERS,ESTIMATORS,A,T,U,N,BATCHES,CAP,FIELDS
from mu_plotting import figure_for_series,audit_rows

def v2_review(summary,args):
    model=summary['model'];frames=N*BATCHES
    rate={.5:'1/2',.25:'1/4',.125:'1/8'}[args.rate]
    text=f"""# نتائج المحاكاة المرجعية V2

تدرس هذه المحاكاة MU downlink بقناة TDL-C و512-QAM وLDPC، بمعدل حمولة مستهدف {rate}. تُقارن MR وZF وPZF كلٌّ على حدة، مع LS وLMMSE وCNN وPerfect CSI. للمحطة 32 أو64 هوائيًا وثماني سلاسل RF، وتخدم أربعة مستخدمين، لكل منهم أربعة هوائيات استقبال وطبقة واحدة.

بقيت السرعة 120km/h، وانتشار التأخير 1µs، والتردد 28GHz، وSNR مراجع الرفع 15dB. يمتد SNR التنزيل من 0 إلى20dB، بخطوة 0.5 حتى10 ثم1dB. المحور هو SNR الاسمي قبل مكاسب المصفوفة والجمع؛ لا يعني 0dB عليه أن SINR بعد الجمع يساوي صفرًا.

تستخدم المقارنات بتات الحمولة نفسها والقناة والضوضاء نفسها. تتغير الموجة المرسلة مع أوزان precoding. مصفوفة RF ثابتة ومتصلة جزئيًا. يستعمل PZF هنا سعة ثابتة وطورًا من4بتات، مع جولتي بحث إحداثي. هذا تقريب مقيد، وليس hybrid phased-ZF المنشور ببنية RF متكيفة يتبعها ZF رقمي.

شملت التصحيحات توحيد نافذة FFT مع مرجع القناة عند CP fraction=0.75، وأوزان baseband لكل حاملة، وتقدير اضطراب مكاني من المراجع المعروفة. يعيد المستقبل فك الحمولات الفاشلة باستعمال حزمة SRS؛ يعتمد قبول المحاولة على CRC، دون معرفة بتات الحمولة الأصلية عند الاختيار.

تضم شبكة المحطة خمسة متنبئات قناة وتتعلم المزج بينها. توجد ثلاثة مستقبلات CNN منفصلة لتصحيح القناة وعدم يقينها. أعداد التدريب والتحقق الفعلية محفوظة في ملخص النموذج. استعمل التدريب رقعًا محلية من24حاملة على12RB، ثم استُخدم النموذج على66RB. لا تدخل القناة الحقيقية أو السرعة الحقيقية إلى الاستدلال العملي.

اختيرت معاملات CNN على16إطار تحقق حمولة مستقل قبل الاختبار. اختير إلغاء تصحيح المحطة واستعمال تصحيح المستقبل؛ لذلك يعني منحنى CNN ‏LMMSE عند BS وCNN عند UE. تحسن MSE شبكة المحطة لم يكن كافيًا لقبولها بحسب BER.

لكل حجم محطة {frames} إطار قناة مستقل، و{frames*U*A} بت حمولة لكل نقطة. يُحسب BER بعد LDPC من جميع الحمولات، بما فيها الحمولات الفاشلة. يُحسب Goodput من بتات الحمولات الصحيحة المجتازة CRC مقسومًا على زمن الإطارات الكامل، مع sounding والحراسة والمراجع. سقف الموارد {CAP:g}Mbit/s. لا يتضمن الحساب MAC أوHARQ. التعديل 512-QAM وموارد sounding تجريبيان ضمن هيكل مبني على NR.

إذا رُصد صفر أخطاء، توضع علامة مفتوحة عند 1/Nbits لظهورها على المحور اللوغاريتمي؛ تحفظ CSV القيمة الأصلية صفرًا. Goodput صفري يعني عدم تسليم حمولة صحيحة ضمن العينة. لا تحل قيمة صفر محل قياس مفقود، ولا ترتب قيم BER لإجبارها على التناقص. المفتاح داخل الرسم وتظهر تدريجات المحور العلمية.
"""
    if args.rate!=.5:
        text+='\nهذه نقطة تشغيل منفصلة بحماية LDPC أقوى وسقف حمولة أقل. بقيت القناة والتعديل والمستخدمون والمراجع وأوزان CNN المجمدة نفسها. اختيرت معاملات CNN مسبقًا بتحقق حمولة عند1/2، ولم تُعد اختيارها على اختبار الحماية الأقوى.\n'
    text+='\n## القيم عند 20dB\n\n| BS | Precoder | BER LS | BER LMMSE | BER CNN | BER Perfect CSI | Goodput CNN Mbit/s |\n|---|---|---|---|---|---|---|\n'
    for array,info in summary['arrays'].items():
        for row in info['points']:
            if row['snr']!=20:continue
            b=row['BER'];text+=f"| {array} | {row['precoder']} | {b['LS']:.6g} | {b['LMMSE']:.6g} | {b['CNN']:.6g} | {b['Perfect CSI']:.6g} | {row['Goodput_Mbps']['CNN']:.6g} |\n"
    text+='\n## معاملات CNN المجمدة\n\n| Precoder | معامل المحطة من تحقق الحمولة | معامل المستقبل من تحقق الحمولة | معامل المستقبل من تحقق القناة |\n|---|---|---|---|\n'
    for precoder in PRECODERS:
        a=model['Selection'][precoder]
        text+=f"| {precoder} | {a[0]:g} | {a[1]:g} | {model['BlendRX'][precoder]:.6g} |\n"
    text+=f"\nمعامل شبكة المحطة من تحقق القناة: {model['BlendTX']:.6g}. المعامل الفعلي هو حاصل ضرب معاملي تحقق القناة والحمولة.\n"
    text+='\nيعرض ملف المقارنة الزوجية فروق CNN وLMMSE وفترات ثقة نقطية بإعادة أخذ العينات على إطار القناة كاملًا. تحجب الفترة عند اختلاف أقل من خمسة أطر؛ لا تصحح هذه الفترات تعدد المقارنات عبر SNR. راجع V2-summary.json لمصدر الدفعات وبصمات الشبكات، وReference-Equation-Audit.md لمراجعة المصادر.\n'
    if args.recovery_run_id:
        text+='\nتوقفت تسع دفعات أصلية عند حد الخادم90دقيقة. أُعيدت كل دفعة في جزأين من8أطر بالمصدر الأصلي والبذور والأوزان نفسها، ثم جُمعت في ترتيب الأطر الأصلي مع التحقق من بصمات الشبكات. لا يُحذف إطار متعثر أو يُعامل كقياس صفري.\n'
        text+=f'\nتشغيل استكمال الدفعات: https://github.com/Seraj21/cod3-matlab/actions/runs/{args.recovery_run_id}\n'
    text+=f'\nالمصدر: https://github.com/Seraj21/cod3-matlab/tree/{args.source}\n\nالتشغيل: https://github.com/Seraj21/cod3-matlab/actions/runs/{args.run_id}\n'
    return text

def v3_review(summary,args):
    model=summary['model'];frames=N*BATCHES
    text=f"""# مراجعة نتائج V3: تقدير القناة الفعالة بالطيار المتاح

هذه محاكاة زمنية لـMU downlink بقناة TDL-C و512-QAM وLDPC، بمعدل حمولة مستهدف 1/8. تُدرس MR وZF وPZF بصورة مستقلة، مع LS وLMMSE وCNN وPerfect CSI. المحطتان 32 و64 هوائيًا، مع ثماني سلاسل RF وأربعة مستخدمين؛ لكل مستخدم أربعة هوائيات استقبال وطبقة واحدة.

بقيت الموارد المرسلة وقدرة المراجع ثابتتين. أصبحت 22 ملاحظة متاحة لكل رقعة محلية من 24 حاملة بدل 18، بإدخال PTRS المرسلة أصلًا مع DMRS إلى LMMSE وCNN. اختير استيفاء LS على تحقق حمولة مستقل؛ يوضح الجدول عدد الملاحظات التي يستخدمها فعليًا. أُعيد حساب إحصاءات LMMSE وتدريب CNN على الملاحظات نفسها. بقيت السرعة 120km/h وانتشار التأخير 1µs والتردد 28GHz وSNR الرفع 15dB، مع تباعد 60kHz وCP ممتد واختبار على 66RB. محور SNR هو الاسمي للتنزيل من 0 إلى 20dB؛ تبقى مكاسب المصفوفة والجمع داخل الأداء، وليس هذا SINR بعد الجمع.

احتفظ MR وZF بأوزان V2. يستعمل PZF بحث SLNR واعيًا بضوضاء التصميم وعدم يقين CSI، ثم تثبيت مرجع طور الاستجابة المرغوبة على شبكة 4 بتات، مع سعة ثابتة وقدرة كلية واحدة. هذا تعريف PZF التجريبي هنا، وليس استنساخًا لخوارزمية hybrid phased-ZF المنشورة. التفاصيل والحدود في V2-Phase-Reference.md.

لكل مستقبل CNN من ست كتل بعرض 48 وبوابتي انتباه خفيفتين. استُخدم الجذع السابق مع إعادة تدريب ورأس خرج ابتداؤه صفر. يتعلم تصحيح LMMSE وعدم يقينه؛ ليس ResNeSt-50. أعداد التدريب والتحقق الفعلية محفوظة في ملخص النموذج. استُخدمت 640 قناة تدريب و80 قناة تحقق لكل مستقبل، برقع محلية على 12RB ثم استدلال على 66RB.

تبدأ بذور القنوات الجديدة للتدريب من 17 مليونًا، والتحقق من 18 مليونًا، وتحقق الحمولة المنفصل من 19 مليونًا. اختير معامل استقبال واحد من 0 أو0.5 أو1 لكل precoder على ثمانية أطر تحقق، قبل الاختبار. تبدأ بذور الاختبار من 50 مليونًا، مع {BATCHES} دفعات من {N} أطر لكل حجم محطة. لا تدخل القناة الحقيقية إلى المقدرات العملية أثناء الاختبار. مرجع Perfect CSI يستعمل القناة والتوقيت الحقيقيين، مع بقاء تشوهات RF في الموجة.

منحنى CNN يعني LMMSE عند BS وتصحيح CNN عند UE بالمعامل المختار. بقي تصحيح شبكة المحطة ملغىً بعد رفضه في تحقق الحمولة السابق. إذا اختير معامل استقبال صفر، يتطابق المنحنيان عن قصد، ويُذكر ذلك في الجدول. لا يُدّعى تفوق مضمون على MMSE البايزي ذي الإحصاءات الصحيحة.

تحتوي النتائج على {frames} إطار قناة مستقل لكل محطة، و{frames*U*A} بت حمولة لكل نقطة. BER محسوب بعد LDPC من جميع الحمولات، بما فيها الفاشلة. Goodput محسوب من الحمولات الصحيحة المجتازة CRC مقسومًا على زمن الإطارات الكامل، مع sounding والحراسة والمراجع. سقفه {CAP:g}Mbit/s، دون MAC أوHARQ. المعدل 1/8 نقطة تشغيل بحماية أقوى وسقف حمولة أقل؛ تحفظ نتائج 1/2 في حزمة مستقلة.

الصفر المرصود في BER يُرسم بعلامة مفتوحة عند 1/Nbits لظهوره على المحور اللوغاريتمي؛ القيمة الأصلية صفر في CSV. Goodput صفري يعني عدم تسليم حمولة صحيحة ضمن العينة. لا توجد تسوية أو إجبار للمنحنيات على الانخفاض.

## القيم عند 20dB

| BS | Precoder | BER LS | BER LMMSE | BER CNN | BER Perfect CSI | Goodput CNN Mbit/s |
|---|---|---|---|---|---|---|
"""
    for array,info in summary['arrays'].items():
        for row in info['points']:
            if row['snr']!=20:continue
            b=row['BER'];text+=f"| {array} | {row['precoder']} | {b['LS']:.6g} | {b['LMMSE']:.6g} | {b['CNN']:.6g} | {b['Perfect CSI']:.6g} | {row['Goodput_Mbps']['CNN']:.6g} |\n"
    text+='\n## معاملات CNN المجمدة\n\n| Precoder | معامل المحطة | معامل المستقبل من تحقق الحمولة | معامل المستقبل من تحقق القناة |\n|---|---|---|---|\n'
    for precoder in PRECODERS:
        a=model['Selection'][precoder]
        text+=f"| {precoder} | {a[0]:g} | {a[1]:g} | {model['BlendRX'][precoder]:.6g} |\n"
    if 'LSInterpolation' in model:
        text+='\n## مقارنة LS المصححة\n\n| Precoder | الاستيفاء المختار بتحقق الحمولة | الملاحظات المستخدمة من22متاحة |\n|---|---|---|\n'
        for precoder in PRECODERS:
            text+=f"| {precoder} | {model['LSInterpolation'][precoder]} | {model['LSReferencesUsed'][precoder]} |\n"
        text+='\nكشف التحكم المستقل أن استبدال الاستيفاء التنسوري باستيفاء مثلثي عالمي أضعف LS. اختُبرت ثلاثة خيارات مسبقة: استيفاء DMRS التنسوري، وتصحيحه بابتكار PTRS، والاستيفاء المثلثي الكامل. أعيد حساب تباين LS من إحصاءات التدريب ومعايرته على80قناة تحقق لكلprecoder، ثم اختير خيار واحد لكلprecoder على أطر تحقق الحمولة الثمانية قبل الاختبار. إذا اختير tensor DMRS، تتلقى الطريقة22ملاحظة متاحة وتستخدم18منها فعليًا؛ لا يُخفى هذا الفرق. تبقى موارد الإرسال متطابقة.\n'
        text+='\nاحتُفظ بتمثيل السمات الذي تدربت عليه CNN، وأثبت اختبار مستقل أن تغيير خط LS وحده لا يغير خرج LMMSE أوCNN. أُعيد تشغيل LS وحده على بذور الاختبار نفسها. تحتوي الحزمة على المكونات الأصلية والمصححة، ويطابق المحلل جميع الأعداد غير التابعة لـLS حرفيًا مع أصلها، ثم يطابق أعداد LS مع تشغيله المصحح. لا يعاد استخدام قياس من قناة أو إطار مختلف.\n'
    text+='\nيعرض V3-paired-CNN-LMMSE.csv الفروق الزوجية وفترات ثقة نقطية بإعادة أخذ العينات على إطار القناة كاملًا. تحجب الفترة إذا اختلفت أقل من خمسة أطر. لا تصحح الفترات تعدد المقارنات عبر SNR. يربط Reference-Equation-Audit.md المعادلات بالتطبيق، ويحفظ V3-summary.json بصمات النماذج ومصدر كل دفعة.\n'
    text+=f'\nالمصدر: https://github.com/Seraj21/cod3-matlab/tree/{args.source}\n\nالتشغيل: https://github.com/Seraj21/cod3-matlab/actions/runs/{args.run_id}\n'
    return text

def audit_configuration(path,array,batch,rate,model,e,f,version=2,seedbase=None):
    with zipfile.ZipFile(path) as archive:
        members=[n for n in archive.namelist() if Path(n).name=='mu_results.mat']
        assert len(members)==1
        result=loadmat(io.BytesIO(archive.read(members[0])),variable_names=['result'],simplify_cells=True)['result']
    expected={'Antennas':array,'RFChains':8,'Users':4,'UERx':4,'NumRB':66,'M':512,'Qm':9,
        'Rate':rate,'Speed':120,'DelaySpread':1e-6,'CarrierFrequency':28e9,
        'SoundingSNR':15,'BeamWidth':1,'CPFraction':.75,'PhaseBits':4,
        'VerticalCorrelation':.7,'HorizontalCorrelation':.3,'UECorrelation':.25,
        'SubcarrierSpacing':60,'InputBackoff':10,'AMPMDegrees':3,'FrequencyOffset':2800,
        'DecoderDual':True,'AlignPhase':True}
    for key,value in expected.items():
        assert result['Config'][key]==value,(path.name,key,result['Config'][key],value)
    assert result['NumPackets']==N and result['PayloadBits']==A and result['CodedBits']==23760
    assert abs(result['FrameDuration']-T)<1e-15
    if seedbase is None:seedbase=20000000 if rate==.5 else 30000000
    assert result['SeedOffset']==seedbase+batch*100000
    if version==3:
        assert result['Config']['Version']==6
        assert result['Config']['PilotMode']=='DMRS+PTRS'
        assert result['Config']['ReferenceObservationsPerLocalChannel']==22
        assert result['Config']['PZFMode']=='robust SLNR, quantized desired-phase anchor'
        assert result['PilotMode']=='DMRS+PTRS'
        assert result['ObservedReferencesPerLocalChannel']==22
        assert model['Version']==6 and model['TargetCodeRate']==rate
        assert model['ReferencesPerLocalChannel']==22
        assert model['AttentionGatesPerReceiver']==2
    np.testing.assert_array_equal(result['Config']['UplinkSymbols'],[1,2,3,4])
    np.testing.assert_array_equal(result['Config']['DownlinkSymbols'],np.arange(6,13))
    np.testing.assert_array_equal(result['Config']['PhaseNoiseLevel'],[-85,-100,-115])
    np.testing.assert_array_equal(result['Config']['PhaseNoiseOffset'],[1e4,1e5,1e6])
    np.testing.assert_array_equal(result['SNR'],SNRS)
    np.testing.assert_array_equal(result['SlotErrors'],e)
    np.testing.assert_array_equal(result['CRCFailed'],f)
    assert np.max(result['PowerError'])<1e-8
    for precoder in PRECODERS:
        np.testing.assert_array_equal(result['ModelSelection'][precoder],model['Selection'][precoder])
        if 'LSInterpolation' in model:
            assert result['LSInterpolation'][precoder]==model['LSInterpolation'][precoder]
            assert result['LSReferencesUsed'][precoder]==model['LSReferencesUsed'][precoder]

def main():
    global A,CAP,N,BATCHES
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',type=int,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--recovery-run-id',type=int)
    parser.add_argument('--version',type=int,choices=[2,3],default=2)
    parser.add_argument('--frames-per-batch',type=int,default=16)
    parser.add_argument('--batches',type=int,default=8)
    parser.add_argument('--seed-base',type=int)
    parser.add_argument('--model-artifact',default='v2-frozen-model.zip')
    parser.add_argument('--model-summary',default='v2_model_summary.json')
    parser.add_argument('--rate',type=float,default=.5,choices=[.5,.25,.125])
    parser.add_argument('--artifact-prefix',default='v2-evaluation')
    parser.add_argument('--archive-name',default='MU-V2-Corrected-CNN-Results.zip')
    args=parser.parse_args();root=args.artifacts;out=args.output;out.mkdir(parents=True,exist_ok=True)
    N=args.frames_per_batch;BATCHES=args.batches
    assert N>0 and BATCHES>0
    raw_audit.N=N;raw_audit.BATCHES=BATCHES
    if args.seed_base is None:args.seed_base=50000000 if args.version==3 else (20000000 if args.rate==.5 else 30000000)
    prefix=f'V{args.version}'
    catalog=json.loads((root/'GitHub-artifact-catalog.json').read_text())
    by_name={item['name']+'.zip':item for item in catalog}
    for path in root.glob('*.zip'):
        assert path.name in by_name, ('No source artifact metadata',path.name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==by_name[path.name]['sha256'], ('Source artifact digest mismatch',path.name)
    shutil.copy2(root/'GitHub-artifact-catalog.json',out/'GitHub-artifact-catalog.json')
    A=8*int(args.rate*23760/8);CAP=U*A/T/1e6
    raw_audit.A=A;raw_audit.CAP=CAP
    rate_label={.5:'1/2',.25:'1/4',.125:'1/8'}[args.rate]
    with zipfile.ZipFile(root/args.model_artifact) as frozen:
        names=[name for name in frozen.namelist() if Path(name).name==args.model_summary]
        assert len(names)==1
        model=json.loads(frozen.read(names[0]))
        smoke=[n for n in frozen.namelist() if n.endswith('mu_results.mat')]
        assert len(smoke)==2
        reference_fingerprints={raw_audit.fingerprint(loadmat(io.BytesIO(frozen.read(n)),variable_names=['cnnFingerprint'])['cnnFingerprint']) for n in smoke}
        assert len(reference_fingerprints)==1
        reference_fingerprint=reference_fingerprints.pop()
    fingerprints=set();allrows=[];paired=[]
    summary={'version':args.version,'run_id':args.run_id,'source':args.source,'model':model,'target_code_rate':args.rate,
             'payload_bits_per_user':A,'maximum_goodput_Mbps':CAP,'arrays':{},
             'frames_per_batch':N,'batches_per_array':BATCHES,'seed_base':args.seed_base,
             'recovery_run_id':args.recovery_run_id,'batch_provenance':[]}
    raw=out/'raw';raw.mkdir(exist_ok=True)
    rng=np.random.default_rng(925117)
    weights=rng.multinomial(N*BATCHES,np.full(N*BATCHES,1/(N*BATCHES)),size=10000)/(N*BATCHES)
    for array in [32,64]:
        es=[];fs=[]
        for batch in range(BATCHES):
            path=root/f'{args.artifact_prefix}-{array}-{batch}.zip'
            e,f,digest=read_batch(path,array);es.append(e);fs.append(f);fingerprints.add(digest)
            audit_configuration(path,array,batch,args.rate,model,e,f,args.version,args.seed_base)
            with zipfile.ZipFile(path) as sourcezip:
                member=next(n for n in sourcezip.namelist() if Path(n).name=='mu_results.mat')
                metadata=loadmat(io.BytesIO(sourcezip.read(member)),variable_names=['result'],simplify_cells=True)['result']
            recovered='Recovery' in metadata
            provenance=by_name[path.name]
            assert provenance['run_id']==(args.recovery_run_id if recovered else args.run_id)
            if not recovered:assert provenance['source']==args.source
            if 'LSRevision' in metadata:
                revision=metadata['LSRevision'];assert args.version==3
                assert revision['ChangedEstimator']=='LS' and revision['LSRun']==args.run_id
                assert revision['RevisionSource']==args.source
                assert revision['OriginalEvaluationRun']==model['OriginalFrozenRun']==41943264421
                assert revision['OriginalSource']=='4f2da4aaf5c685c06b0425affe630eeb69cedec8'
                assert revision['SelectionSeedBase']==19000000 and revision['ChannelCalibrationSeedBase']==18000000
                original=root/f'v3-evaluation-{array}-{batch}.zip'
                revised_ls=root/f'v3-ls-evaluation-{array}-{batch}.zip'
                oe,of,oh=read_batch(original,array)
                le,lf,lh=read_batch(revised_ls,array,estimators=['LS'])
                np.testing.assert_array_equal(e[:,:,:,1:,:],oe[:,:,:,1:,:])
                np.testing.assert_array_equal(f[:,:,:,1:,:],of[:,:,:,1:,:])
                np.testing.assert_array_equal(e[:,:,:,:1,:],le)
                np.testing.assert_array_equal(f[:,:,:,:1,:],lf)
                assert oh==lh==digest
                assert by_name[original.name]['run_id']==revision['OriginalEvaluationRun']
                assert by_name[original.name]['source']==revision['OriginalSource']
                assert by_name[revised_ls.name]['run_id']==args.run_id
                assert by_name[revised_ls.name]['source']==args.source
                for component in [original,revised_ls]:
                    destination=out/'provenance-components'/component.name
                    destination.parent.mkdir(exist_ok=True);shutil.copy2(component,destination)
                summary.setdefault('LS_revision',revision)
            if recovered:
                assert args.rate==.5 and args.recovery_run_id
                assert (array,batch) in [(32,2),(32,3),(32,4),(32,5),(32,6),(64,1),(64,2),(64,5),(64,6)]
                np.testing.assert_array_equal(metadata['Recovery']['PartFrames'],[8,8])
                np.testing.assert_array_equal(metadata['Recovery']['PartSeedOffsets'],[20000000+batch*100000,20000008+batch*100000])
                assert metadata['Recovery']['OriginalSource']==args.source
            summary['batch_provenance'].append({'BSAntennas':array,'batch':batch,'frames':N,
                'run_id':args.recovery_run_id if recovered else args.run_id,
                'recovered_from_exact_seed_subdivisions':recovered,
                'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
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
                    out/f'{prefix}-{precoder}-{kind}-BS{array}.png',
                    measured_label=f'V{args.version} | Independent frozen-model simulation | LDPC target rate {rate_label}\nCNN curve: selected LMMSE at BS + learned CNN at UE')
        summary['arrays'][str(array)]={'frames':len(e),'payload_bits_per_point':bits,'points':points,
            'CNN_BER_better':int(np.count_nonzero(ber[:,:,2]<ber[:,:,1])),
            'CNN_BER_worse':int(np.count_nonzero(ber[:,:,2]>ber[:,:,1])),
            'CNN_BER_equal':int(np.count_nonzero(ber[:,:,2]==ber[:,:,1])),
            'CNN_Goodput_better':int(np.count_nonzero(gp[:,:,2]>gp[:,:,1]))}
    assert fingerprints=={reference_fingerprint},'Evaluation network weights differ from the frozen model smoke tests'
    summary['network_fingerprint']=fingerprints.pop()
    audit=audit_rows(allrows,SNRS);summary['plot_audit']=audit
    writecsv(out/f'{prefix}-all-metrics.csv',allrows,FIELDS)
    writecsv(out/f'{prefix}-paired-CNN-LMMSE.csv',paired)
    writecsv(out/f'{prefix}-plot-audit.csv',audit)
    manifest=[]
    for path in sorted(root.glob('*.zip')):
        if args.version==2 and path.name.startswith('v3-'):continue
        if path.name.startswith(('v2-evaluation-','v2-robust-','v3-evaluation-','v3-ls-evaluation-','v3-final-evaluation-')) and not path.name.startswith(args.artifact_prefix+'-'):
            continue
        if path.name.startswith('v2-half-part-'):continue
        manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if path.name==args.model_artifact:shutil.copy2(path,out/path.name)
        elif not path.name.startswith(args.artifact_prefix+'-'):
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if Path(member).suffix in ['.csv','.json']:
                        target=out/'training'/path.stem/member
                        target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(archive.read(member))
    for name in ['V2-Reference-Sources.md','V2-Method-Review.md','V2-Phase-Reference.md','Reference-Equation-Audit.md']:
        source=Path(__file__).resolve().parent.parent/name
        if source.exists():shutil.copy2(source,out/name)
    summary['artifacts']=manifest
    summary['checks']=['All local artifact SHA256 hashes match independently retrieved GitHub artifact metadata',
        f'All {2*BATCHES} evaluation ZIP archives passed CRC validation',
        'Every CSV/MAT BER and goodput reproduced from individual user payload errors and CRC flags',
        'Every per-user CSV error and CRC flag agrees with the full MAT tensor',
        'Every MAT configuration, independent test seed, coding rate and precoder power matches the declared experiment',
        '31 SNR values for each of 24 estimator curves, with no missing or duplicate measurement',
        f'Four identical frozen CNN weight sets in all {2*BATCHES} batches',
        'No BER smoothing, sorting, forced monotonicity or substituted zero measurements']
    (out/f'{prefix}-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    text=v3_review(summary,args) if args.version==3 else v2_review(summary,args)
    (out/f'{prefix}-Review.md').write_text(text,encoding='utf-8')
    archive=out.parent/args.archive_name
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['artifacts','plot_audit']},ensure_ascii=False))
    print('Archive:',archive,'SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())

if __name__=='__main__':main()
