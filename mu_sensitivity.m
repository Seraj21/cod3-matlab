function mu_sensitivity
addpath('mu_mimo');loaded=load('models/mu_cnn.mat','model');model=loaded.model;
for array=[32 64]
 cfg=muConfig(array,DelaySpread=300e-9);r=muResources(cfg);
 result=muBenchmark(cfg,r,model,2,[0 4 8 12 16 20],4400000);
 folder=fullfile("sensitivity","nominal"+array);muExport(result,model,folder);
end
cfg=muConfig(64);r=muResources(cfg);
rows=cell(0,7);
for p=1:2
 scene=muSounding(cfg,r,4500000+p);
 [grid,payload]=muPayload(cfg,r,muSeed(4500000+p,70));
 for width=[4 12 24]
  local=scene;local.Config.PRGWidth=width;resource=r;resource.G=r.K/width;
  for scheme=["MR" "ZF" "PZF"]
   w=muWeights(scene.ULTruth,scheme,local,resource);
   packet=muDownlink(scene,r,grid,w);
   input=muReceive(packet,scene,r,20,true);
   [e,f]=muDecode(input,input.Truth,zeros(size(input.Truth)),payload,scene.Config,r);
   rows(end+1,:)={p,width,scheme,20,sum(e),numel(payload),nnz(f)};
  end
 end
end
T=cell2table(rows,'VariableNames',{'Packet','PrecodingSubcarriers','Precoder','SNR_dB','PayloadBitErrors','PayloadBits','FailedBlocks'});
writetable(T,fullfile("sensitivity","precoder_width.csv"));disp(T);
end
