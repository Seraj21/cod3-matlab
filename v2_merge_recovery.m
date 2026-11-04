function v2_merge_recovery(root,outputRoot,modelFile)
% Reassemble exact original frame indices after timeout-only subdivision.
addpath('mu_mimo','v2_mimo');
loaded=load(modelFile,'model');model=loaded.model;
pairs=[32 2;32 3;32 4;32 5;32 6;64 1;64 2;64 5;64 6];
for k=1:size(pairs,1)
 bs=pairs(k,1);batch=pairs(k,2);base=20000000+batch*100000;
 parts=cell(1,2);fingerprints=cell(1,2);
 for part=0:1
  name=sprintf('v2-recovery-%d-%d-%d',bs,batch,part);
  a=load(fullfile(root,name,'mu_results.mat'),'result','cnnFingerprint');
  parts{part+1}=a.result;fingerprints{part+1}=a.cnnFingerprint;
  assert(a.result.NumPackets==8 && a.result.SeedOffset==base+8*part);
  assert(a.result.Config.Antennas==bs && a.result.Config.Rate==.5);
  assert(isequal(a.result.SNR,[0:.5:10 11:20]));
 end
 assert(isequal(parts{1}.Config,parts{2}.Config));
 assert(isequal(parts{1}.ModelSelection,parts{2}.ModelSelection));
 assert(isequal(fingerprints{1},fingerprints{2}));
 result=parts{1};
 for field=["SlotErrors" "CRCFailed" "SRSBeamAccepted" "PowerError"]
  result.(field)=cat(1,parts{1}.(field),parts{2}.(field));
 end
 result.NumPackets=16;result.SeedOffset=base;
 result.Recovery=struct('PartFrames',[8 8],'PartSeedOffsets',[base base+8], ...
  'OriginalSource','ec00ae08f0e470071158768d92b8320c490784c3');
 folder=fullfile(outputRoot,sprintf('v2-evaluation-%d-%d',bs,batch));
 v2Export(result,model,folder);
 final=load(fullfile(folder,'mu_results.mat'),'cnnFingerprint');
 assert(isequal(final.cnnFingerprint,fingerprints{1}));
 fprintf('Reassembled BS%d batch%d: original seeds %d to %d\n',bs,batch,base+1,base+16);
end
end
