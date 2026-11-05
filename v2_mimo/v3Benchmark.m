function result=v3Benchmark(cfg,r,model,frames,snrDb,seedOffset,folder)
% Genuine waveform comparison, with per-frame checkpoints and identical noise.
if nargin<7,folder="";end
schemes=["MR" "ZF" "PZF"];names=["LS" "LMMSE" "CNN" "Perfect CSI"];internal=["LS" "LMMSE" "CNN" "Perfect"];
errors=zeros(frames,numel(snrDb),3,4,4);failed=false(size(errors));rescued=failed;
powerError=zeros(frames,3,4);
for frame=1:frames
 scene=v2Sounding(cfg,r,seedOffset+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 state=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
 for p=1:3
  scheme=schemes(p);rx=model.Receivers.(scheme);selected=model.Selection.(scheme);assert(selected(1)==0);
  rx.BlendRX=rx.BlendRX*selected(2);
  for method=1:4
   if method==4,csit=scene.ULTruth;uncertainty=zeros(size(csit));
   elseif method==1,csit=state.LS;uncertainty=state.VarianceLS;
   else,csit=state.Wiener;uncertainty=state.VarianceWiener;end
   [weights,d]=v3Weights(csit,uncertainty,scheme,scene,r);powerError(frame,p,method)=d.MaximumPowerError;
   packet=v2Waveform(scene,r,grid,weights,method==4);
   for s=1:numel(snrDb)
    input=v3Receive(packet,scene,r,snrDb(s),method==4,false);
    [h,v,observed]=v2EstimateRX(input,rx,r,internal(method));input.Observed=observed;
    [e,f,detail]=v2Decode(input,h,v,payload,cfg,r);
    errors(frame,s,p,method,:)=reshape(e,1,1,1,1,4);failed(frame,s,p,method,:)=reshape(f,1,1,1,1,4);
    rescued(frame,s,p,method,:)=reshape(detail.SRSBeamAccepted,1,1,1,1,4);
   end
  end
 end
 result=struct('Config',cfg,'NumPackets',frame,'SNR',snrDb,'Schemes',schemes,'Estimators',names, ...
  'EstimatorIndices',1:4,'SlotErrors',errors(1:frame,:,:,:,:),'CRCFailed',failed(1:frame,:,:,:,:), ...
  'SRSBeamAccepted',rescued(1:frame,:,:,:,:),'PayloadBits',r.PayloadBits,'CodedBits',r.CodedBits, ...
  'FrameDuration',r.FrameDuration,'SeedOffset',seedOffset,'PowerError',powerError(1:frame,:,:), ...
  'ModelSelection',model.Selection,'PilotMode',"DMRS+PTRS",'ObservedReferencesPerLocalChannel',22);
 if strlength(folder)>0,v2Export(result,model,folder);end
 fprintf('V3 checkpoint: BS%d frame%d/%d seed%d complete\n',cfg.Antennas,frame,frames,scene.Seed);
end
end
