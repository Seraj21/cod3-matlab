# إعادة تشغيل المقارنة

تحتاج MATLAB R2025b مع Communications Toolbox و5G Toolbox وDeep Learning Toolbox. نُفذت التجربة على CPU. تحتوي حزمة النماذج المجمدة على الشبكات وإحصاءات القناة ومعاملات الاختيار؛ لا يلزم إعادة التدريب لإعادة الاختبار.

ضع `MU-V3-Frozen-CNN-Models.zip` في جذر المستودع، ثم:

```matlab
unzip('MU-V3-Frozen-CNN-Models.zip','v3_refined');
snrDb = [0:.5:10 11:20];
for bs = [32 64]
    for batch = 0:7
        folder = fullfile('local_v3',sprintf('BS%d_batch%d',bs,batch));
        run_v3_mimo(bs,4,folder,snrDb,50000000+batch*100000, ...
            'v3_refined/v3_model.mat');
    end
end
```

هذا يعيد البذور المنشورة: ثماني دفعات من أربعة أطر لكل حجم محطة، بمجموع32إطارًا. تشغيل32إطارًا متتابعًا ببذرة واحدة تجربة صالحة، لكنه لا يستخدم قائمة البذور الموزعة نفسها.

يشغّل الأمر MR وZF وPZF، ولكل منها LS وLMMSE وCNN وPerfect CSI. تحفظ كل دفعة:

- `mu_metrics.csv`: ‏BER الحمولة بعد LDPC وGoodput عند كل SNR.
- `mu_slot_errors.csv`: أخطاء كل مستخدم ونتيجة CRC، دون حذف الحمولات الفاشلة.
- `mu_results.mat`: الأعداد الكاملة والإعدادات وبصمات أوزان الشبكات.

تُكتب نقطة تحقق بعد اكتمال كل إطار. يعكس `NumPackets` الأطر المكتملة فقط؛ لا تُملأ بقية النتائج بأصفار إذا توقف التشغيل.

النظام: محطة32أو64هوائيًا، وثماني سلاسلRF، وأربعة مستخدمين، لكل منهم أربعة هوائيات استقبال وطبقة واحدة. القناة TDL-C، والسرعة120km/h، وانتشار التأخير1µs، والتردد28GHz، والرفعSNR=15dB، والتعديل512-QAM، ومعدل حمولة LDPC المستهدف1/8. يبقى مرجع1/2 في حزمة مستقلة. سقف Goodput لمعدل1/8 هو47.488Mbit/s، ويشمل الزمن sounding والحراسة والمراجع.

يستعمل منحنى CNN ‏LMMSE في المحطة وتصحيح CNN في المستقبل بالمعامل المختار على تحقق مستقل. تصحيح المحطة ملغى بعد رفضه في تحقق الحمولة السابق. يستعمل Perfect CSI أيضًا التوقيت الحقيقي. لا يُدخل المرجع الحقيقي إلى الطرق العملية عند الاستدلال.

تتاح22ملاحظة طيار محلية من DMRS وPTRS المرسلتين أصلًا. يحتفظ LMMSE وCNN بتمثيل الاستيفاء المطابق لتدريبهما. يوضح ملخص النموذج استيفاء LS المختار وعدد الملاحظات التي يستخدمها فعليًا؛ اختير قبل الاختبار من ثلاثة خيارات، بعد معايرة تباينه على قنوات تحقق مستقلة.

للتحقق من بيانات GitHub المحفوظة، يستعمل محلل Python ‏NumPy وSciPy وMatplotlib، ويتحقق من بصمات الملفات والأعداد وCSV/MAT ومصدر كل مكون. المثال المخصص للحزمة النهائية:

```bash
python3 tools/analyze_v2.py --version 3 --frames-per-batch 4 \
  --seed-base 50000000 --rate .125 \
  --artifacts artifacts --output reviewed_v3 \
  --run-id 41946464299 \
  --source a0583f0cd2347a9e0a93bfbc873a6ce623648fb6 \
  --artifact-prefix v3-final-evaluation \
  --model-artifact v3-refined-frozen-model.zip \
  --model-summary v3_model_summary.json \
  --archive-name MU-V3-Reviewed-Results.zip
```

لإعادة تدقيق الحزمة المنشورة، اجمع ZIP الموجودة في raw وprovenance-components مع v3-refined-frozen-model.zip وGitHub-artifact-catalog.json في مجلد artifacts. يشير الأمر إلى هذه الملفات الأصلية وسجل بصماتها كما جُمعت من GitHub. لا يُستخدم لقبول ملفات محلية جديدة على أنها الملفات الأصلية نفسها. لإعادة إنتاج الأعداد محليًا، قارن جداول CSV الناتجة بأعداد الحزمة المنشورة، مع نفس إصدار MATLAB والنموذج والبذور.

المراجعة العلمية في [Reference-Equation-Audit.md](Reference-Equation-Audit.md)، وتعريف مرجع طور PZF وحدوده في [V2-Phase-Reference.md](V2-Phase-Reference.md). لا تُفرض استقامة أو رتابة المنحنيات، ولا يُفسر صفر أخطاء مرصودًا على أنه ضمان لخطأ صفري.
