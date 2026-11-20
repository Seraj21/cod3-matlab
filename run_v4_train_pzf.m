function run_v4_train_pzf(folder,useProjection,modelFile,trainFrames,validationFrames,epochs,scheme,matchedPilots,scaledResidual,densePilots)
% Fit the receiver on the requested deployment waveform and pilot coordinates.
if nargin<3,modelFile='v3_refined/v3_model.mat';end
if nargin<4,trainFrames=128;end
if nargin<5,validationFrames=32;end
if nargin<6,epochs=10;end
if nargin<7,scheme="PZF";end
if nargin<8,matchedPilots=false;end
if nargin<9,scaledResidual=false;end
if nargin<10,densePilots=false;end
assert(~scaledResidual || (matchedPilots && useProjection));
scheme=string(scheme);
numRB=66;if matchedPilots,numRB=264;end
assert(isscalar(scheme) && ismember(scheme,["MR" "ZF" "PZF"]));
addpath(pwd,'mu_mimo','v2_mimo');
loaded=load(modelFile,'model');model=loaded.model;
assert(model.Version==6 && model.Selection.(scheme)(1)==0);
if ~isfolder(folder),mkdir(folder);end
cfg=v3Config(32,numRB,"mobile",matchedPilots,densePilots);r=v3Resources(cfg);r.ScaledResidual=scaledResidual;r.DensePilots=densePilots;times=cfg.DownlinkSymbols;
% Refit the deployed transmitter's predictor before collecting receiver data.
% These channel seeds are disjoint from receiver training, validation and test.
if scaledResidual && ~densePilots
 base=load(fullfile('prior_receiver',lower(scheme)+"_receiver.mat"),'receiver');
 assert(base.receiver.MatchedPilotMask && base.receiver.TrainingBandwidthRB==numRB);
 model.PriorsTX=base.receiver.PriorsTX;txFrames=base.receiver.TransmitterPriorChannels;
else
txStats=v2Prior("init",r.UplinkLayout,0);
txFrames=64;if matchedPilots,txFrames=24;end
for frame=1:txFrames
 txCfg=v3Config(32+32*mod(frame-1,2),numRB,"mobile",matchedPilots,densePilots);
 scene=v2Sounding(txCfg,r,97000000+frame);
 txStats=v2Prior("accumulate",txStats,scene.ULCleanSamples,scene.ULTruth,r.UplinkLayout);
end
model.PriorsTX=v2Prior("finalize",txStats);clear txStats scene
end
layouts=cell(1,8);
for user=1:4
 layout=r.DownlinkLayout{user};
 layouts{user}=layout;layouts{user}.Label="Own_"+layout.Label;
 layouts{user+4}=layout;layouts{user+4}.Label="Other_"+layout.Label;
end
stats=v2Prior("init",layouts,4);
[training,stats,trainConditions]=collect(model,r,trainFrames,72000000,stats,true,scheme);
[validation,~,valConditions]=collect(model,r,validationFrames,73000000,[],false,scheme);
conditions=[trainConditions;valConditions];
writetable(conditions,fullfile(folder,'conditions.csv'));
priors=v2Prior("finalize",stats);clear stats
% The tuned and matched LMMSE receivers share the same prior and pilot inputs.
[V,Y]=examples(validation,priors,r,useProjection);
clear validation
priors=calibrateVariance(priors,Y,[],0,times);
[X,T]=examples(training,priors,r,useProjection);
clear training
receiver=model.Receivers.(scheme);receiver.PriorsRX=priors;
% Use the same transmitted references for LS and the matched estimators.
legacyFields=intersect(fieldnames(receiver),{'LSInterpolation','PriorsLS'});
if ~isempty(legacyFields),receiver=rmfield(receiver,legacyFields);end
net=receiver.NetworkRX;
loss=@v2RXLoss;
if useProjection,loss=@v4AlignedLoss;end
amplitude=1;
if scaledResidual
 % Learn the small conditional correction relative to the matched Wiener
 % estimate; keep its calibrated error covariance as the variance baseline.
 times=unique(r.DataTimes);
 amplitude=max(1e-4,sqrt(mean(T(:,times,1:128,:).^2,'all')));
 net=v3RXNetwork(size(X,3),16,net);
 T(:,:,1:128,:)=T(:,:,1:128,:)/amplitude;
 Y(:,:,1:128,:)=Y(:,:,1:128,:)/amplitude;
 loss=@(z,t)conditionalMSE(z,t,times);
end
rng(721600,'twister');
frequency=max(1,ceil(size(X,4)/32));
options=trainingOptions('adam','MaxEpochs',epochs,'MiniBatchSize',32, ...
 'InitialLearnRate',1e-4,'LearnRateSchedule','piecewise','LearnRateDropPeriod',6, ...
 'LearnRateDropFactor',.5,'L2Regularization',1e-5,'GradientThreshold',1, ...
 'Shuffle','every-epoch','ExecutionEnvironment','cpu','ValidationData',{V,Y}, ...
 'ValidationFrequency',frequency,'ValidationPatience',5,'OutputNetwork','best-validation', ...
 'Verbose',true,'VerboseFrequency',frequency);
fprintf('%s projection=%d: %d train patches, %d validation patches\n', ...
 scheme,useProjection,size(X,4),size(V,4));
[net,info]=trainnet(X,T,net,loss,options);
clear X T
if scaledResidual
 % Convert the fitted head back into the native channel normalization so
 % inference uses the same operator as validation and the existing receiver.
 values=net.Learnables;
 head=find(string(values.Layer)=="mean_and_log_variance");
 assert(numel(head)==2);
 for k=head.'
  values.Value{k}=values.Value{k}*amplitude;
 end
 net=dlupdate(@(~,value)value,net,values);
 Y(:,:,1:128,:)=Y(:,:,1:128,:)*amplitude;
end
writetable(info.TrainingHistory,fullfile(folder,'learning.csv'));
predicted=zeros(6,12,192,size(V,4),'single');
for k=1:32:size(V,4)
 idx=k:min(k+31,size(V,4));
 predicted(:,:,:,idx)=extractdata(predict(net,dlarray(V(:,:,:,idx),'SSCB')));
end
times=cfg.DownlinkSymbols;if scaledResidual,times=unique(r.DataTimes);end
target=double(Y(:,times,1:128,:));residual=double(predicted(:,times,1:128,:));
blend=max(0,min(1,sum(residual.*target,'all')/max(sum(residual.^2,'all'),eps)));
priors=calibrateVariance(priors,Y,predicted,blend,times);
receiver.PriorsRX=priors;receiver.NetworkRX=net;receiver.BlendRX=blend;receiver.Signature=model.Signature;
receiver.Signature.TargetCodeRate=cfg.Rate;
receiver.TrainingChannels=trainFrames;receiver.ValidationChannels=validationFrames;
receiver.TrainingPatches=24*trainFrames*(1+3*scaledResidual);receiver.TrainingResidualScale=amplitude;receiver.ProjectionLoss=useProjection;
receiver.TrainingBandwidthRB=numRB;receiver.TrainingCodeRate=cfg.Rate;
receiver.TrainingSNR_dB=[min(conditions.SNR1_dB) max(conditions.SNR2_dB)];
if matchedPilots
 receiver.PilotMasks=cellfun(@(p)p.Mask,r.DownlinkLayout,'UniformOutput',false);
 receiver.MatchedPilotMask=true;receiver.DensePilots=densePilots;
 receiver.DownlinkSymbols=cfg.DownlinkSymbols;receiver.UplinkSymbols=cfg.UplinkSymbols;
end
receiver.PriorsTX=model.PriorsTX;
receiver.TransmitterPriorChannels=txFrames;receiver.TransmitterPriorSeedBase=97000000;
% Prevent silently deploying a receiver trained for another channel/RF profile.
receiver.TrainingChannelConfig=struct;
for name=["OperatingProfile" "DelaySpread" "Speed" "SoundingSNR" "CarrierFrequency" "SubcarrierSpacing" ...
 "PhaseNoiseLevel" "PhaseNoiseOffset" "InputBackoff" "AMPMDegrees" "FrequencyOffset" ...
 "VerticalCorrelation" "HorizontalCorrelation" "UECorrelation" "CPFraction" "BeamWidth" "PhaseBits" "PrecoderTimeMode"]
 receiver.TrainingChannelConfig.(name)=cfg.(name);
end
receiver.TrainingSeedBase=72000000;receiver.ValidationSeedBase=73000000;
receiver.ValidationNormalizedMSE=[mean(target.^2,'all') mean((target-blend*residual).^2,'all')];
save(fullfile(folder,lower(scheme)+"_receiver.mat"),'receiver','-v7.3');
fprintf('%s validation blend %.6f; normalized MSE %.6g -> %.6g\n', ...
 scheme,blend,receiver.ValidationNormalizedMSE);
end
function [records,stats,conditions]=collect(model,r,count,seedBase,stats,training,scheme)
crops=1+3*r.ScaledResidual;records=cell(8*count*crops,1);at=0;conditions=zeros(count,5);
for frame=1:count
 is64=mod(frame-1,2);bs=32+32*is64;
 cfg=v3Config(bs,r.K/12,"mobile",r.MatchedPilotMask,r.DensePilots);seed=seedBase+frame;
 scene=v2Sounding(cfg,r,seed);
 tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
 weights=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
 [grid,~]=muPayload(cfg,r,muSeed(seed,70));
 packet=v2Waveform(scene,r,grid,weights,true);
 n=ceil(frame/2)-1;
 points=[.5*mod(8*n+4*is64,21) 11+mod(7*n+3*is64,30)];
 if r.MatchedPilotMask
  points=[15+5*mod(n+is64,4),35+5*mod(n+is64,2)];
 end
 conditions(frame,:)=[training seed bs points];
 for snr=points
  input=v3Receive(packet,scene,r,snr,false,training);
  observed=v2BeamObserved(input.Observed,r);
  for user=1:4
   [samples,truth,layouts,order]=v2RXCanonical(input,r,user);
   if training
    clean=v2RXCanonical(input,r,user,true);
    stats=v2Prior("accumulate",stats,clean,truth,layouts);
   end
   rows=(user-1)*4+(1:4);
   for crop=1:crops
   first=1+mod(7*(frame-1)+5*(user-1)+(crop-1)*floor(r.G/crops),r.G-6);
   tones=(first-1)*24+(1:168);
   at=at+1;
   records{at}=struct('Samples',single(samples(tones,:,:,:)), ...
    'Truth',single(truth(tones,:,:,:)),'Layouts',{layouts},'Order',order, ...
    'Noise',input.Noise(rows),'Observed',single(observed(tones,:,rows)), ...
    'Tones',tones);
   end
  end
 end
 if mod(frame,8)==0 || frame==count
  fprintf('%s training=%d collected %d/%d channels\n',scheme,training,frame,count);
 end
end
conditions=array2table(conditions,'VariableNames',{'Training','ChannelSeed','BS','SNR1_dB','SNR2_dB'});
end
function [X,T]=examples(records,priors,r,useProjection)
targetChannels=192+128*useProjection;
X=zeros(6,12,584,3*numel(records),'single');
T=zeros(6,12,targetChannels,3*numel(records),'single');
for k=1:numel(records)
 item=records{k};state=v2CEBase(item.Samples,item.Noise,item.Layouts,priors);
 [x,y,meta]=v2RXFeatures(state,item.Truth,item.Layouts,r);
 if useProjection
  % The selected center bands have the same covariance window as deployment.
  local=struct('ReferenceGrid',r.ReferenceGrid(item.Tones,:,item.Order));
  g=permute(state.Wiener,[3 4 1 2]);
  disturbance=v2NoiseCovariance(double(item.Observed),state.Wiener, ...
   state.VarianceWiener,item.Noise,local);
  w=pagemldivide(pagemtimes(g,'none',g,'ctranspose')+disturbance,g(:,1,:,:));
  w=permute(reshape(w,4,meta.K,meta.L),[2 3 1]);
  scale=reshape(repelem(state.Scale,meta.W,1,1),meta.K,1,meta.R,meta.U);
  coefficient=conj(w).*scale;
  coefficient=coefficient./max(sqrt(sum(abs(coefficient).^2,3)),1e-12);
  y=cat(3,y,v2ParameterPatches(real(coefficient),meta),v2ParameterPatches(imag(coefficient),meta));
 end
 idx=3*(k-1)+(1:3);X(:,:,:,idx)=x(:,:,:,3:5);T(:,:,:,idx)=y(:,:,:,3:5);
end
end
function priors=calibrateVariance(priors,targets,predictions,blend,times)
% Receiver reliability is estimated from a separate channel split.
keys=string(fieldnames(priors));
for role=["Own" "Other"]
 layers=1;if role=="Other",layers=2:4;end
 for branch=1:4
  indices=reshape((1:4).'+4*(branch-1)+16*(layers-1),1,[]);
  er=double(targets(:,times,indices,:));ei=double(targets(:,times,64+indices,:));
  v=exp(double(targets(:,times,128+indices,:)));
  if isempty(predictions)
   factor=max(1e-4,sum(er.^2+ei.^2,'all')/max(sum(v,'all'),eps));
   method='LMMSE';
  else
   er=er-blend*double(predictions(:,times,indices,:));
   ei=ei-blend*double(predictions(:,times,64+indices,:));
   v=v.*exp(max(-5,min(5,double(predictions(:,times,128+indices,:)))));
   factor=max(1e-4,sum(er.^2+ei.^2,'all')/max(sum(v,'all'),eps));
   method='CNN';
  end
  selected=keys(startsWith(keys,role+"_") & endsWith(keys,"_R"+branch));
  for key=selected.'
   priors.(key).VarianceFactor.(method)=factor;
  end
 end
end
end

function loss=conditionalMSE(z,t,times)
% The first 128 channels are 64 packed complex errors. Target channels
% 193:320 encode the receiver's normalized projection for each stream.
er=z(:,times,1:64,:)-t(:,times,1:64,:);
ei=z(:,times,65:128,:)-t(:,times,65:128,:);
cr=t(:,times,193:256,:);ci=t(:,times,257:320,:);
realPart=er.*cr-ei.*ci;imagPart=er.*ci+ei.*cr;
idx=reshape((1:4).'+16*(0:3),1,[]);
pr=realPart(:,:,idx,:)+realPart(:,:,idx+4,:)+realPart(:,:,idx+8,:)+realPart(:,:,idx+12,:);
pi=imagPart(:,:,idx,:)+imagPart(:,:,idx+4,:)+imagPart(:,:,idx+8,:)+imagPart(:,:,idx+12,:);
loss=mean(er.^2+ei.^2,'all')+.5*mean(pr.^2+pi.^2,'all');
end
