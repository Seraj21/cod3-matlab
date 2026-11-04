function receiver=v2TrainRX(scheme,txFile,folder,epochs,maximumFrames)
% Recreate deterministic channels locally; only small trained models are exported.
if nargin<4,epochs=40;end
if nargin<5,maximumFrames=Inf;end
scheme=string(scheme);
if ~isfolder(folder),mkdir(folder);end
loaded=load(txFile,'model');transmitter=loaded.model;clear loaded
conditions=readtable(fullfile(fileparts(txFile),'v2_conditions.csv'));
names=["MR" "ZF" "PZF"];which=find(names==scheme);assert(isscalar(which));
conditions=conditions(conditions.Scheme==which,:);
if isfinite(maximumFrames)
 a=find(conditions.Training);b=find(~conditions.Training);
 conditions=conditions([a(1:min(maximumFrames,numel(a)));b(1:min(8,numel(b)))],:);
end
writetable(conditions,fullfile(folder,'rx_conditions.csv'));
cfg=v2Config(32,12);r=muResources(cfg);
layouts=roleLayouts(r);stats=v2Prior("init",layouts,4);
nTrain=nnz(conditions.Training)*8;nValid=nnz(~conditions.Training)*8;
training=cell(nTrain,1);validation=cell(nValid,1);a=0;b=0;
for index=1:height(conditions)
 row=conditions(index,:);cfg=v2ConfigFromRow(row,12);scene=v2Sounding(cfg,r,row.Seed);
 sources=["LS" "LMMSE" "CNN" "Perfect"];
 source=sources(row.CSITSource);
 if source=="Perfect",csit=scene.ULTruth;
 else,csit=v2EstimateTX(scene.ULSamples,scene.ULNoise,transmitter,r,source);end
 weights=v2Weights(csit,scheme,scene,r);
 [grid,~]=muPayload(cfg,r,muSeed(row.Seed,70));
 packet=v2Waveform(scene,r,grid,weights,true);
 for snr=[row.DataSNR1 row.DataSNR2]
  input=v2Receive(packet,scene,r,snr,false,true);
  for user=1:4
   [samples,truth,localLayouts]=v2RXCanonical(input,r,user);
   [clean,~,~]=v2RXCanonical(input,r,user,true);
   % The same random UE oscillator phase rotates observations and labels.
   stream=RandStream('mt19937ar','Seed',muSeed(muSeed(row.Seed,81,user),round(100*snr)));
   phase=exp(1i*2*pi*rand(stream));samples=samples*phase;truth=truth*phase;clean=clean*phase;
   if row.Training,stats=v2Prior("accumulate",stats,clean,truth,localLayouts);end
   item=v2CropRecord(samples,truth,input.Noise((user-1)*4+(1:4)),row.Frame,row.Seed,r);
   item.User=user;
   if row.Training,a=a+1;training{a}=item;else,b=b+1;validation{b}=item;end
  end
 end
 if mod(index,32)==0,fprintf('%s RX collected %d/%d independent channels\n',scheme,index,height(conditions));end
end
priors=v2Prior("finalize",stats);clear stats input scene packet samples truth clean
[X,T]=examples(training,priors,r);clear training
[V,Y]=examples(validation,priors,r);
rng(730001+which,'twister');net=v2RXNetwork(size(X,3),16);
frequency=max(1,ceil(size(X,4)/32));
options=trainingOptions('adam','MaxEpochs',epochs,'MiniBatchSize',32, ...
 'InitialLearnRate',2e-4,'LearnRateSchedule','piecewise','LearnRateDropPeriod',10, ...
 'LearnRateDropFactor',.5,'L2Regularization',1e-5,'GradientThreshold',1, ...
 'Shuffle','every-epoch','ExecutionEnvironment','cpu','ValidationData',{V,Y}, ...
 'ValidationFrequency',frequency,'ValidationPatience',16,'OutputNetwork','best-validation', ...
 'Verbose',true,'VerboseFrequency',frequency);
fprintf('Training %s RX: %d train patches, %d held-out patches, %d epochs maximum\n', ...
 scheme,size(X,4),size(V,4),epochs);
[net,info]=trainnet(X,T,net,@v2RXLoss,options);
clear X T V Y
try,writetable(info.TrainingHistory,fullfile(folder,'rx_learning.csv'));
catch,save(fullfile(folder,'rx_learning.mat'),'info');end
[priors,blend,summary]=calibrate(net,validation,priors,r);
writetable(summary,fullfile(folder,'rx_channel_validation.csv'));
receiver=struct('Version',5,'Scheme',scheme,'PriorsRX',priors,'NetworkRX',net, ...
 'BlendRX',blend,'TrainingChannels',nnz(conditions.Training), ...
 'ValidationChannels',nnz(~conditions.Training),'TrainingPatches',nTrain*2, ...
 'Signature',transmitter.Signature);
save(fullfile(folder,'v2_rx_'+lower(scheme)+'.mat'),'receiver','-v7.3');
fprintf('%s independent channel-validation blend %.6f\n',scheme,blend);
end
function list=roleLayouts(r)
list=cell(4,1);at=0;
for role=["Own" "Other"]
 for index=[1 3]
  at=at+1;p=r.DownlinkLayout{index};p.Label=role+"_"+p.Label;list{at}=p;
 end
end
end
function layouts=forUser(r,user)
order=[user setdiff(1:4,user,'stable')];layouts=r.DownlinkLayout(order);
for u=1:4
 role="Other";if u==1,role="Own";end
 layouts{u}.Label=role+"_"+layouts{u}.Label;
end
end
function [X,T]=examples(records,priors,r)
G=size(records{1}.Samples,1)/24;N=numel(records)*G;
X=zeros(6,12,584,N,'single');T=zeros(6,12,192,N,'single');
for index=1:numel(records)
 item=records{index};layouts=forUser(r,item.User);
 state=v2CEBase(item.Samples,item.Noise,layouts,priors);
 [x,t]=v2RXFeatures(state,item.Truth,layouts,r);
 selected=(index-1)*G+(1:G);X(:,:,:,selected)=x;T(:,:,:,selected)=t;
end
end
function [priors,blend,summary]=calibrate(net,records,priors,r)
numerator=0;denominator=0;rows=zeros(numel(records),5);
% A single global coefficient is fitted on independent channels, never test BER.
for index=1:numel(records)
 [state,update,~]=prediction(records{index},priors,r,net);
 truth=double(records{index}.Truth(:,6:12,:,:));e=truth-state.Wiener(:,6:12,:,:);
 d=update(:,6:12,:,:);numerator=numerator+real(sum(conj(d).*e,'all'));
 denominator=denominator+sum(abs(d).^2,'all');
end
blend=max(0,min(1,numerator/max(denominator,eps)));
keys=string(fieldnames(priors));sums=zeros(numel(keys),6);
for index=1:numel(records)
 item=records{index};[state,update,variance]=prediction(item,priors,r,net);
 truth=double(item.Truth);value=state.Wiener+blend*update;layouts=forUser(r,item.User);
 rows(index,:)=[item.Seed item.User mean(abs(truth(:,6:12,:,:)-state.LS(:,6:12,:,:)).^2,'all') ...
  mean(abs(truth(:,6:12,:,:)-state.Wiener(:,6:12,:,:)).^2,'all') ...
  mean(abs(truth(:,6:12,:,:)-value(:,6:12,:,:)).^2,'all')];
 for u=1:4
  for branch=1:4
   k=find(keys==layouts{u}.Label+"_R"+branch);t=truth(:,6:12,branch,u);
   sums(k,:)=sums(k,:)+[sum(abs(t-state.LS(:,6:12,branch,u)).^2,'all') ...
    sum(abs(t-state.Wiener(:,6:12,branch,u)).^2,'all') ...
    sum(abs(t-value(:,6:12,branch,u)).^2,'all') ...
    sum(state.RawVarianceLS(:,6:12,branch,u),'all') ...
    sum(state.RawVarianceWiener(:,6:12,branch,u),'all') sum(variance(:,6:12,branch,u),'all')];
  end
 end
end
for k=1:numel(keys)
 priors.(keys(k)).VarianceFactor=struct('LS',max(1e-4,sums(k,1)/max(sums(k,4),eps)), ...
 'LMMSE',max(1e-4,sums(k,2)/max(sums(k,5),eps)), ...
 'CNN',max(1e-4,sums(k,3)/max(sums(k,6),eps)));
end
summary=array2table(rows,'VariableNames',{'ChannelSeed','User','MSE_LS','MSE_LMMSE','MSE_CNN'});
end
function [state,update,variance]=prediction(item,priors,r,net)
layouts=forUser(r,item.User);state=v2CEBase(item.Samples,item.Noise,layouts,priors);
[X,~,meta]=v2RXFeatures(state,[],layouts,r);
z=double(extractdata(predict(net,dlarray(X,'SSCB'))));
[update,variance]=v2RXPrediction(z,state,meta);
end
