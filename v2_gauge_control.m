function v2_gauge_control(bs)
% Literature-driven Heff diagnostic. All operational estimates ignore true CSI.
addpath(pwd,'mu_mimo','v2_mimo');load('v2_frozen/v2_model.mat','model');
cfg=v2Config(bs);cfg.Rate=.125;r=muResources(cfg);rows=cell(0,19);
rx=model.Receivers.PZF;rx.BlendRX=rx.BlendRX*model.Selection.PZF(2);
assert(model.Selection.PZF(1)==0);
variantNames=["Baseline W1" "Robust W1" "Robust W4" "Quantized anchor W1" ...
 "Quantized anchor W4" "Digital phase anchor W1" "Digital phase anchor W4"];
for frame=1:2
 scene=v2Sounding(cfg,r,6800000+bs*1000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 state=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
 for source=["LS" "LMMSE" "Perfect"]
  if source=="Perfect",csit=scene.ULTruth;variance=zeros(size(csit));methods="Perfect";
  elseif source=="LS",csit=state.LS;variance=state.VarianceLS;methods="LS";
  else,csit=state.Wiener;variance=state.VarianceWiener;methods=["LMMSE" "CNN"];end
  local=scene;base1=v2Weights(csit,"PZF",local,r);robust1=v2RobustPhaseWeights(csit,variance,local,r,base1);
  local.Config.BeamWidth=4;base4=v2Weights(csit,"PZF",local,r);robust4=v2RobustPhaseWeights(csit,variance,local,r,base4);
  candidates={base1,robust1,robust4,[],[],[],[]};diagnostics=cell(1,7);
  for variant=[4 5 6 7]
   width=1;if mod(variant,2)==1,width=4;end
   local.Config.BeamWidth=width;base=robust1;if width==4,base=robust4;end
   mode="quantized";if variant>=6,mode="continuous";end
   [candidates{variant},diagnostics{variant}]=v2PhaseGauge(base,csit,local,r,mode);
  end
  for variant=1:7
   width=1;if any(variant==[3 5 7]),width=4;end
   local.Config.BeamWidth=width;weights=candidates{variant};packet=v2Waveform(local,r,grid,weights,true);
   powerError=0;latticeError=0;
   if variant>=4,powerError=diagnostics{variant}.TotalPowerError;latticeError=diagnostics{variant}.RelativePhaseLatticeError;end
   for snr=[8 12 16 20]
    input=v2Receive(packet,local,r,snr,source=="Perfect",false);
    truth=v2EstimateRX(input,rx,r,"Perfect");
    for method=methods
     [h,v,observed]=v2EstimateRX(input,rx,r,method);decodedInput=input;decodedInput.Observed=observed;
     [e,f]=v2Decode(decodedInput,h,v,payload,cfg,r);
     rows=record(rows,frame,bs,variantNames(variant),source,method,false,snr,e,f, ...
      ownNMSE(h,truth,r),powerError,latticeError,scene.Seed,r);
    end
    if source~="Perfect" && snr==20
     [h,v,observed]=v2EstimateRX(input,rx,r,"Perfect");decodedInput=input;decodedInput.Observed=observed;
     [e,f]=v2Decode(decodedInput,h,v,payload,cfg,r);
     rows=record(rows,frame,bs,variantNames(variant),source,"Perfect",true,snr,e,f,0,powerError,latticeError,scene.Seed,r);
    end
   end
  end
 end
 T=cell2table(rows,'VariableNames',{'Frame','BSAntennas','Variant','CSIT','Estimator','PerfectRXControl', ...
  'SNR_dB','Errors','Bits','CorrectBlocks','Goodput_Mbps','OwnDataChannelNMSE','PowerError', ...
  'RelativePhaseLatticeError','ChannelSeed','TargetCodeRate','FrameDuration','PerUserErrors','PerUserCRCFailed'});
 writetable(T,'v2_gauge_control.csv');
 fprintf('Gauge control BS%d frame%d/2 complete\n',bs,frame);
end
end
function rows=record(rows,frame,bs,variant,source,method,oracle,snr,e,f,nmse,power,lattice,seed,r)
correct=nnz(e==0 & ~f);
rows(end+1,:)={frame,bs,variant,source,method,oracle,snr,sum(e),4*r.PayloadBits,correct, ...
 correct*r.PayloadBits/r.FrameDuration/1e6,nmse,power,lattice,seed,.125,r.FrameDuration,jsonencode(e),jsonencode(f)};
end
function nmse=ownNMSE(h,truth,r)
a=0;b=0;
for u=1:4
 row=(u-1)*4+1;x=h(:,:,row,u);y=truth(:,:,row,u);
 a=a+sum(abs(x(r.DataIndices)-y(r.DataIndices)).^2);b=b+sum(abs(y(r.DataIndices)).^2);
end
nmse=a/max(b,eps);
end
