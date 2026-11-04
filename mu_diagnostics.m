function mu_diagnostics
addpath('mu_mimo');
loaded=load('models/mu_cnn.mat','model');model=loaded.model;
cases=["quiet_static" "quiet_fading" "rf_static" "full_hard"];
for k=1:numel(cases)
 quiet=any(k==[1 2]);dynamic=any(k==[2 4]);
 cfg=muConfig(64,Quiet=quiet);
 if ~dynamic,cfg.Speed=0;cfg.DelaySpread=30e-9;end
 r=muResources(cfg);
 result=muBenchmark(cfg,r,model,2,[0 10 20 40],4200000,["MR" "ZF" "PZF"],4);
 folder=fullfile("diagnostics",cases(k));muExport(result,model,folder);
 disp(cases(k));disp(readtable(fullfile(folder,"mu_metrics.csv")));
end
end
