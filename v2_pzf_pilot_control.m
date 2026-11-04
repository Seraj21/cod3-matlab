function v2_pzf_pilot_control
% Diagnose quantized-precoder roughness versus DM-RS CDM/interpolation assumptions.
addpath('mu_mimo','v2_mimo');load('v2_frozen/v2_model.mat','model');
cfg=v2Config(64);cfg.Rate=.125;r=muResources(cfg);rows=cell(0,15);
methods=["LS" "LMMSE" "CNN" "Perfect"];rx=model.Receivers.PZF;
rx.BlendRX=rx.BlendRX*model.Selection.PZF(2);assert(model.Selection.PZF(1)==0);
for frame=1:2
 scene=v2Sounding(cfg,r,6700000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 state=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
 for method=1:4
  if method==4,csit=scene.ULTruth;variance=zeros(size(csit));
  elseif method==1,csit=state.LS;variance=state.VarianceLS;
  else,csit=state.Wiener;variance=state.VarianceWiener;end
  for width=[1 4]
   local=scene;local.Config.BeamWidth=width;
   baseline=v2Weights(csit,"PZF",local,r);
   robust=v2RobustPhaseWeights(csit,variance,local,r,baseline);
   for variant=1:2
    weights=baseline;if variant==2,weights=robust;end
    smooth=mean(sum(abs(diff(double(weights(:,:,:,cfg.DownlinkSymbols)),1,3)).^2,[1 2]),'all');
    packet=v2Waveform(local,r,grid,weights,true);
    for snr=[8 12 16 20]
     input=v2Receive(packet,local,r,snr,method==4,false);
     [truth,~,~]=v2EstimateRX(input,rx,r,"Perfect");
     pristine=sum(input.Truth.*reshape(r.ReferenceGrid,r.K,r.L,1,4),4);
     fake=input;fake.Samples=muPilotObservations(pristine,r,"DL");
     ideal=v2EstimateRX(fake,rx,r,"LS");
     [h,v,observed]=v2EstimateRX(input,rx,r,methods(method));input.Observed=observed;
     [e,f]=v2Decode(input,h,v,payload,cfg,r);correct=nnz(e==0 & ~f);
     name="Baseline PZF";if variant==2,name="Robust phase-only control";end
     rows(end+1,:)={frame,width,name,methods(method),snr,sum(e),numel(payload),correct, ...
      correct*r.PayloadBits/r.FrameDuration/1e6,ownNMSE(h,truth,r),ownNMSE(ideal,truth,r), ...
      smooth,cfg.Rate,cfg.Antennas,scene.Seed}; %#ok<AGROW>
    end
   end
  end
 end
 T=cell2table(rows,'VariableNames',{'Frame','BeamWidth','Variant','Estimator','SNR_dB','Errors', ...
  'Bits','CorrectBlocks','Goodput_Mbps','OwnDataChannelNMSE','NoiselessPilotLS_NMSE', ...
  'FrequencyWeightRoughness','TargetCodeRate','BSAntennas','ChannelSeed'});
 writetable(T,'v2_pzf_pilot_control.csv');
 fprintf('Completed phase/pilot control frame%d/2\n',frame);
end
fprintf('V2_PZF_PILOT_CONTROL_BEGIN\n%s\nV2_PZF_PILOT_CONTROL_END\n',jsonencode(table2struct(T)));
end
function nmse=ownNMSE(h,truth,r)
numerator=0;denominator=0;
for u=1:4
 row=(u-1)*4+1;a=h(:,:,row,u);b=truth(:,:,row,u);
 numerator=numerator+sum(abs(a(r.DataIndices)-b(r.DataIndices)).^2);
 denominator=denominator+sum(abs(b(r.DataIndices)).^2);
end
nmse=numerator/max(denominator,eps);
end
