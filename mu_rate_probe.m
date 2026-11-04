function mu_rate_probe
addpath('mu_mimo');loaded=load('models/mu_cnn.mat','model');model=loaded.model;
rows=cell(0,8);
for rate=[.25 .125]
 cfg=muConfig(64);cfg.Rate=rate;r=muResources(cfg);
 result=muBenchmark(cfg,r,model,2,[0 4 8 12 16 20],4800000,"ZF");
 for m=1:4
  for s=1:numel(result.SNR)
   e=squeeze(result.SlotErrors(:,s,1,m,:));
   f=squeeze(result.CRCFailed(:,s,1,m,:));
   success=nnz(e==0 & ~f);
   rows(end+1,:)={rate,r.PayloadBits,result.Estimators(m),result.SNR(s),sum(e,'all')/numel(e)/r.PayloadBits,success*r.PayloadBits/(2*r.FrameDuration)/1e6,success,4*r.PayloadBits/r.FrameDuration/1e6};
  end
 end
end
T=cell2table(rows,'VariableNames',{'TargetCodeRate','PayloadBitsPerUE','Estimator','SNR_dB','PostLDPC_BER','Goodput_Mbps','CorrectBlocks','ResourceCeiling_Mbps'});
writetable(T,'rate_probe.csv');
fprintf('MU_RATE_BEGIN\n%s\nMU_RATE_END\n',jsonencode(table2struct(T)));
end
