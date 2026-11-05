function v3_refine_freeze
addpath(pwd,'mu_mimo','v2_mimo');folder='v3_refined';if ~isfolder(folder),mkdir(folder);end
loaded=load('v3_base/v3_model.mat','model');model=loaded.model;assert(model.Version==6);
assert(~isfield(model,'LSInterpolation'));model=v3RefineLS(model,'rx_models',folder);
for bs=[32 64]
 cfg=v3Config(bs);r=v3Resources(cfg);
 v3Benchmark(cfg,r,model,1,[0 10 20],19600000+bs*1000,fullfile(folder,'smoke_BS'+string(bs)));
end
save(fullfile(folder,'v3_model.mat'),'model','-v7.3');
summary=jsondecode(fileread('v3_base/v3_model_summary.json'));
summary.LSInterpolation=model.LSInterpolation;summary.LSReferencesUsed=model.LSReferencesUsed;
summary.LSChannelCalibrationFramesPerPrecoder=80;summary.LSPayloadValidationFrames=8;
summary.LSValidation=model.LSValidation;summary.OriginalFrozenRun=41943264421;
summary.CNNFeatureInterpolation='triangular full pilots, unchanged from matched training';
fid=fopen(fullfile(folder,'v3_model_summary.json'),'w');fprintf(fid,'%s',jsonencode(summary));fclose(fid);
copyfile('v3_base/v3_payload_validation.csv',fullfile(folder,'v3_payload_validation.csv'));
fprintf('V3_REFINED_SUMMARY_BEGIN\n%s\nV3_REFINED_SUMMARY_END\n',jsonencode(summary));
end
