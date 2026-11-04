function v2_operating_point_probe
% Keep the original four-layer/rate-1/2 comparison; diagnose feasible modes separately.
addpath('mu_mimo','v2_mimo');model=v2Assemble('v2_tx_training/v2_tx.mat','rx_models');
rows=cell(0,13);methods=["LS" "LMMSE" "CNN" "Perfect"];
for scenario=1:3
 cfg=v2Config(64);
 if scenario==1,cfg.Rate=.5;activeCount=2;
 elseif scenario==2,cfg.Rate=.25;activeCount=4;
 else,cfg.Rate=.125;activeCount=4;end
 r=muResources(cfg);
 for frame=1:2
  scene=v2Sounding(cfg,r,6500000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
  active=1:4;if activeCount==2,active=[1 3]+mod(frame-1,2);end
  inactive=setdiff(1:4,active);local=scene;local.Config.ActiveUsers=active;
  for scheme=["MR" "ZF" "PZF"]
   rx=model.Receivers.(scheme);
   for method=1:4
    if method==4,csit=scene.ULTruth;
    else,csit=v2EstimateTX(scene.ULSamples,scene.ULNoise,model,r,methods(method));end
    precodingScene=scene;precodingScene.Config.Users=activeCount;
    small=v2Weights(csit(:,:,:,active),scheme,precodingScene,r);
    weights=complex(zeros(8,4,r.K,r.L,'single'));weights(:,active,:,:)=small;
    packet=v2Waveform(scene,r,grid,weights,method==4);
    for snr=[0 8 12 16 20]
     input=v2Receive(packet,scene,r,snr,method==4,false);
     [h,v,observed]=v2EstimateRX(input,rx,r,methods(method));input.Observed=observed;
     h(:,:,:,inactive)=0;v(:,:,:,inactive)=0;
     [e,f]=v2Decode(input,h,v,payload,local.Config,r);
     correct=nnz(e(active)==0 & ~f(active));
     rows(end+1,:)={scenario,cfg.Rate,activeCount,frame,scheme,methods(method),snr, ...
      sum(e(active)),activeCount*r.PayloadBits,correct,correct*r.PayloadBits/r.FrameDuration/1e6, ...
      activeCount*r.PayloadBits/r.FrameDuration/1e6,jsonencode(active)}; %#ok<AGROW>
    end
   end
  end
 end
end
T=cell2table(rows,'VariableNames',{'Scenario','TargetCodeRate','ScheduledUsers','Frame','Precoder', ...
 'Estimator','SNR_dB','Errors','Bits','CorrectBlocks','Goodput_Mbps','ResourceCeiling_Mbps','ActiveUsers'});
writetable(T,'v2_operating_point_probe.csv');
fprintf('V2_OPERATING_POINTS_BEGIN\n%s\nV2_OPERATING_POINTS_END\n',jsonencode(table2struct(T)));
end
