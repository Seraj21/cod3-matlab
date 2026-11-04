function mu_dynamic_probe
addpath('mu_mimo');loaded=load('models/mu_cnn.mat','model');model=loaded.model;
rows=cell(0,10);
for speed=[0 120]
 for ds=[30e-9 1e-6]
  cfg=muConfig(64,NumRB=12,Quiet=true,Speed=speed,DelaySpread=ds);
  r=muResources(cfg);scene=muSounding(cfg,r,4300001);
  [grid,payload]=muPayload(cfg,r,4300002);
  weights=muWeights(scene.ULTruth,"ZF",scene,r);
  packet=muDownlink(scene,r,grid,weights);
  input=muReceive(packet,scene,r,60,true);
  pred=sum(input.Truth.*reshape(grid,r.K,r.L,1,4),4);
  mask=repmat(r.DataMask,1,1,16);
  evm=sum(abs(input.Observed(mask)-pred(mask)).^2)/sum(abs(pred(mask)).^2);
  ch=scene.Channel;reset(ch);
  [~,directH,td]=ch(zeros(r.Samples+scene.Tail,8,'single'),r.Carrier);
  directH=double(directH(:,1:r.L,:,:)).*reshape(scene.Front.ChannelGain,1,1,1,8);
  reciprocalError=sum(abs(directH-scene.HDownlink).^2,'all')/sum(abs(directH).^2,'all');
  H=permute(directH,[3 4 1 2]);
  effective=permute(pagemtimes(H,double(weights)),[3 4 1 2]);
  direct=input.Truth;
  for u=1:4
   cols=(u-1)*4+(1:4);
   direct(:,:,cols,:)=muRFReference(effective(:,:,cols,:),packet.TxPhase,packet.RxPhase{u}, ...
      -input.CFOResidual(u),td,td,r);
  end
  predDirect=sum(direct.*reshape(grid,r.K,r.L,1,4),4);
  evmDirect=sum(abs(input.Observed(mask)-predDirect(mask)).^2)/sum(abs(predDirect(mask)).^2);
  input20=muReceive(packet,scene,r,20,true);
  [e,f]=muDecode(input20,input20.Truth,zeros(size(input20.Truth)),payload,scene.Config,r);
  rows(end+1,:)={speed,ds*1e9,"auto",reciprocalError,evm,evmDirect,sum(e)/numel(payload),nnz(f),mean(input.CFOResidual),td};
  % Native signal-rate path gains, using a direct response of the very same call.
  fresh=muChannel(cfg,scene.Front,r.OFDM.SampleRate,4300001);
  fresh.PathGainSampleRate='signal';
  x=sum(permute(weights,[3 4 1 2]).*reshape(single(grid),r.K,r.L,1,4),4);
  wave=nrOFDMModulate(r.Carrier,x,'Windowing',0);
  [rx,Hsig,tSig]=fresh([wave.*scene.Front.ChannelGain.';zeros(scene.Tail,8,'single')],r.Carrier);
  Hsig=double(Hsig(:,1:r.L,:,:)).*reshape(scene.Front.ChannelGain,1,1,1,8);
  p2=packet;p2.Waveform=double(rx);p2.TrueOffset=tSig;
  p2.Propagation=permute(pagemtimes(permute(Hsig,[3 4 1 2]),double(weights)),[3 4 1 2]);
  p2.TxPhase=ones(size(packet.TxPhase));
  for u=1:4,p2.RxPhase{u}=ones(size(packet.RxPhase{u}));end
  i2=muReceive(p2,scene,r,60,true);
  pred=sum(i2.Truth.*reshape(grid,r.K,r.L,1,4),4);
  evm2=sum(abs(i2.Observed(mask)-pred(mask)).^2)/sum(abs(pred(mask)).^2);
  i20=muReceive(p2,scene,r,20,true);
  [e,f]=muDecode(i20,i20.Truth,zeros(size(i20.Truth)),payload,scene.Config,r);
  rows(end+1,:)={speed,ds*1e9,"signal",NaN,evm2,evm2,sum(e)/numel(payload),nnz(f),mean(i2.CFOResidual),tSig};
 end
end
T=cell2table(rows,'VariableNames',{'Speed_kmh','RMSDelay_ns','PathGainRate','ReciprocityRelativeMSE', ...
 'WaveformRelativeMSE','DirectResponseRelativeMSE','BER20','FailedBlocks20','MeanCFOResidual60','TrueOffset'});
writetable(T,'dynamic_probe.csv');disp(T);
end
