function result=v2Benchmark(cfg,r,model,frames,snrDb,seedOffset,schemes,methods)
if nargin<7,schemes=["MR" "ZF" "PZF"];end
if nargin<8,methods=1:4;end
names=["LS" "LMMSE" "CNN" "Perfect CSI"];internal=["LS" "LMMSE" "CNN" "Perfect"];
errors=zeros(frames,numel(snrDb),numel(schemes),numel(methods),4);
failed=false(size(errors));rescued=false(size(errors));
powerError=zeros(frames,numel(schemes),numel(methods));
for frame=1:frames
 seed=seedOffset+frame;scene=v2Sounding(cfg,r,seed);
 [grid,payload]=muPayload(cfg,r,muSeed(seed,70));
 for p=1:numel(schemes)
  scheme=schemes(p);tx=model;rx=model.Receivers.(scheme);
  selected=model.Selection.(scheme);
  tx.BlendTX=tx.BlendTX*selected(1);rx.BlendRX=rx.BlendRX*selected(2);
  for m=1:numel(methods)
   method=methods(m);
   if method==4,csit=scene.ULTruth;
   else,csit=v2EstimateTX(scene.ULSamples,scene.ULNoise,tx,r,internal(method));end
   [weights,d]=v2Weights(csit,scheme,scene,r);
   powerError(frame,p,m)=d.MaximumPowerError;
   packet=v2Waveform(scene,r,grid,weights,method==4);
   for s=1:numel(snrDb)
    input=v2Receive(packet,scene,r,snrDb(s),method==4,false);
    [h,v,observed]=v2EstimateRX(input,rx,r,internal(method));input.Observed=observed;
    [e,f,detail]=v2Decode(input,h,v,payload,scene.Config,r);
    errors(frame,s,p,m,:)=reshape(e,1,1,1,1,4);
    failed(frame,s,p,m,:)=reshape(f,1,1,1,1,4);
    rescued(frame,s,p,m,:)=reshape(detail.SRSBeamAccepted,1,1,1,1,4);
   end
  end
 end
 fprintf('V2 independent frame %d/%d, BS%d, seed %d complete\n',frame,frames,cfg.Antennas,seed);
end
result=struct('Config',cfg,'NumPackets',frames,'SNR',snrDb,'Schemes',schemes, ...
 'Estimators',names(methods),'EstimatorIndices',methods,'SlotErrors',errors,'CRCFailed',failed, ...
 'SRSBeamAccepted',rescued,'PayloadBits',r.PayloadBits,'CodedBits',r.CodedBits, ...
 'FrameDuration',r.FrameDuration,'SeedOffset',seedOffset,'PowerError',powerError, ...
 'ModelSelection',model.Selection);
end
