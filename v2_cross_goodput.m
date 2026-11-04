function v2_cross_goodput
% Separate transmitter aging from receiver estimation without altering the link.
addpath('mu_mimo','v2_mimo');model=v2Assemble('v2_tx_training/v2_tx.mat','rx_models');
cfg=v2Config(64);r=muResources(cfg);rx=model.Receivers.ZF;rows=cell(0,9);
cases=["Perfect" "Perfect";"LS" "Perfect";"LMMSE" "Perfect";"CNN" "Perfect"; ...
 "Perfect" "LS";"Perfect" "LMMSE";"Perfect" "CNN"];
for frame=1:4
 scene=v2Sounding(cfg,r,6400000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 for index=1:size(cases,1)
  txMethod=cases(index,1);rxMethod=cases(index,2);
  if txMethod=="Perfect",hTX=scene.ULTruth;
  else,hTX=v2EstimateTX(scene.ULSamples,scene.ULNoise,model,r,txMethod);end
  weights=v2Weights(hTX,"ZF",scene,r);packet=v2Waveform(scene,r,grid,weights,true);
  for snr=[12 16 20]
   % Practical timing in every case isolates CSI rather than timing knowledge.
   input=v2Receive(packet,scene,r,snr,false,false);
   [h,v,observed]=v2EstimateRX(input,rx,r,rxMethod);input.Observed=observed;
   [e,f]=v2Decode(input,h,v,payload,scene.Config,r);
   correct=nnz(e==0 & ~f);
   rows(end+1,:)={frame,txMethod,rxMethod,snr,sum(e),numel(payload), ...
       correct,correct*r.PayloadBits/r.FrameDuration/1e6,mean(v,'all')}; %#ok<AGROW>
  end
 end
end
T=cell2table(rows,'VariableNames',{'Frame','CSIT','CSIR','SNR_dB','Errors','Bits', ...
 'CorrectBlocks','Goodput_Mbps','MeanChannelVariance'});writetable(T,'v2_cross_goodput.csv');
fprintf('V2_GOODPUT_CROSS_BEGIN\n%s\nV2_GOODPUT_CROSS_END\n',jsonencode(table2struct(T)));
end
