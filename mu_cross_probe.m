function mu_cross_probe
addpath('mu_mimo');loaded=load('models/mu_cnn.mat','model');model=loaded.model;
rows=cell(0,10);
for ds=[300e-9 1e-6]
 cfg=muConfig(64,DelaySpread=ds);r=muResources(cfg);
 for p=1:2
  scene=muSounding(cfg,r,4600000+p);
  [grid,payload]=muPayload(cfg,r,muSeed(4600000+p,70));
  [~,~,state]=muCE(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX,r,"LS");
  ht={state.Wiener,scene.ULTruth};
  for ti=1:2
   w=muWeights(ht{ti},"ZF",scene,r);packet=muDownlink(scene,r,grid,w);
   for rm=["LMMSE" "CNN" "Perfect CSI"]
    input=muReceive(packet,scene,r,20,rm=="Perfect CSI");
    if rm=="Perfect CSI",h=input.Truth;v=zeros(size(h));
    elseif rm=="CNN"
     [h,v]=muCE(input.Samples,input.Noise,r.DownlinkLayout,model.PriorsRX,r,rm,model.NetworkRX,model.BlendRX*model.BlendMultiplier,4);
    else
     [h,v]=muCE(input.Samples,input.Noise,r.DownlinkLayout,model.PriorsRX,r,rm);
    end
    [e,f,d]=muDecode(input,h,v,payload,scene.Config,r);
    mask=repmat(r.DataMask,1,1,16,4);truth=input.Truth;
    nmse=sum(abs(h(mask)-truth(mask)).^2)/sum(abs(truth(mask)).^2);
    names=["LMMSE" "Perfect CSI"];
    rows(end+1,:)={ds*1e9,p,names(ti),rm,sum(e)/numel(payload),nnz(f),nmse,mean(d.MeanEffectiveVariance),mean(abs(input.CFOResidual)),mean(input.TimingError)};
   end
  end
 end
end
T=cell2table(rows,'VariableNames',{'RMSDelay_ns','Packet','TX_CSI','RX_CSI','BER20','FailedBlocks','ChannelNMSE','MeanLLRVariance','MeanAbsoluteCFOResidual','MeanTimingError'});
writetable(T,'cross_probe.csv');disp(T);
fprintf('MU_CROSS_BEGIN\n%s\nMU_CROSS_END\n',jsonencode(table2struct(T)));
end
