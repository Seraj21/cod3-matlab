function v3_merge_ls
% Replace only the LS measurements using exact matching test seeds.
% All other counts are copied unchanged, with explicit revision provenance.
addpath(pwd,'mu_mimo','v2_mimo');loaded=load('v3_refined/v3_model.mat','model');model=loaded.model;
for bs=[32 64]
 for batch=0:7
  name=sprintf('v3-evaluation-%d-%d',bs,batch);
  original=load(fullfile('original_results',name,'mu_results.mat'));
  revised=load(fullfile('ls_results',sprintf('v3-ls-evaluation-%d-%d',bs,batch),'mu_results.mat'));
  result=original.result;ls=revised.result;
  assert(isequaln(result.Config,ls.Config) && result.NumPackets==4 && ls.NumPackets==4);
  assert(result.SeedOffset==50000000+batch*100000 && ls.SeedOffset==result.SeedOffset);
  assert(isequal(result.SNR,ls.SNR) && isequal(result.SNR,[0:.5:10 11:20]));
  assert(result.PayloadBits==2968 && ls.PayloadBits==result.PayloadBits && result.CodedBits==ls.CodedBits);
  assert(isequaln(original.cnnFingerprint,revised.cnnFingerprint));
  assert(isequaln(result.ModelSelection,ls.ModelSelection) && isequaln(ls.ModelSelection,model.Selection));
  assert(isequal(result.Schemes,ls.Schemes) && isequal(ls.Estimators,"LS"));
  assert(max(abs(result.PowerError(:,:,1)-ls.PowerError(:,:,1)),[],'all')<1e-10);
  result.SlotErrors(:,:,:,1,:)=ls.SlotErrors;
  result.CRCFailed(:,:,:,1,:)=ls.CRCFailed;
  result.SRSBeamAccepted(:,:,:,1,:)=ls.SRSBeamAccepted;
  result.PowerError(:,:,1)=ls.PowerError(:,:,1);
  result.LSInterpolation=model.LSInterpolation;result.LSReferencesUsed=model.LSReferencesUsed;
  result.LSRevision=struct('OriginalEvaluationRun',41943264421, ...
   'OriginalSource','4f2da4aaf5c685c06b0425affe630eeb69cedec8', ...
   'LSRun',str2double(getenv('GITHUB_RUN_ID')),'RevisionSource',getenv('GITHUB_SHA'), ...
   'ChangedEstimator','LS','SelectionSeedBase',19000000,'ChannelCalibrationSeedBase',18000000);
  assert(isequal(result.SlotErrors(:,:,:,2:4,:),original.result.SlotErrors(:,:,:,2:4,:)));
  assert(isequal(result.CRCFailed(:,:,:,2:4,:),original.result.CRCFailed(:,:,:,2:4,:)));
  v2Export(result,model,fullfile('v3_merged',sprintf('v3-final-evaluation-%d-%d',bs,batch)));
 end
end
end
