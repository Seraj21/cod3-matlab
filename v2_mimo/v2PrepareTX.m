function model=v2PrepareTX(folder,trainingFrames,validationFrames,epochs)
if ~isfolder(folder),mkdir(folder);end
conditions=v2Recipe(trainingFrames,validationFrames);
writetable(conditions,fullfile(folder,'v2_conditions.csv'));
cfg=v2Config(32,12);r=muResources(cfg);speeds=[0 30 60 90 120];
general=v2Prior("init",r.UplinkLayout,0);experts=cell(5,1);
for k=1:5,experts{k}=v2Prior("init",r.UplinkLayout,0);end
training=cell(trainingFrames,1);validation=cell(validationFrames,1);
for index=1:height(conditions)
 row=conditions(index,:);cfg=v2ConfigFromRow(row,12);
 scene=v2Sounding(cfg,r,row.Seed);
 item=v2CropRecord(scene.ULSamples,scene.ULTruth,scene.ULNoise,row.Frame,row.Seed,r);
 item.Speed=row.Speed;
 if row.Training
  general=v2Prior("accumulate",general,scene.ULCleanSamples,scene.ULTruth,r.UplinkLayout);
  k=find(speeds==row.Speed);
  experts{k}=v2Prior("accumulate",experts{k},scene.ULCleanSamples,scene.ULTruth,r.UplinkLayout);
  training{row.Frame}=item;
 else,validation{row.Frame}=item;end
 if mod(index,64)==0,fprintf('Collected TX %d/%d independent channels\n',index,height(conditions));end
end
model=struct('Version',5,'PriorsTX',v2Prior("finalize",general),'ExpertsTX',{cell(5,1)}, ...
 'BlendTX',1,'TrainingFrames',trainingFrames,'ValidationFrames',validationFrames,'Speeds',speeds);
for k=1:5,model.ExpertsTX{k}=v2Prior("finalize",experts{k});end
clear general experts scene
[X,E,T,labels]=examples(training,model,r);clear training
[V,VE,VT,Vlabels]=examples(validation,model,r);
[model.NetworkTX,~]=v2GateTrain(X,E,T,labels,V,VE,VT,Vlabels,epochs,folder);
clear X E T labels V VE VT Vlabels
records=zeros(numel(validation),5);numerator=0;denominator=0;
for index=1:numel(validation)
 item=validation{index};state=v2CEBase(item.Samples,item.Noise,r.UplinkLayout,model.PriorsTX);
 [estimate,p]=v2EstimateTX(item.Samples,item.Noise,model,r,"CNN");
 error=double(item.Truth(:,6:12,:,:))-state.Wiener(:,6:12,:,:);
 update=estimate(:,6:12,:,:)-state.Wiener(:,6:12,:,:);
 numerator=numerator+real(sum(conj(update).*error,'all'));
 denominator=denominator+sum(abs(update).^2,'all');
 records(index,:)=[item.Seed item.Speed mean(abs(error).^2,'all') ...
     mean(abs(error-update).^2,'all') mean(speeds*p,'all')];
end
model.BlendTX=max(0,min(1,numerator/max(denominator,eps)));
writetable(array2table(records,'VariableNames',{'ChannelSeed','Speed_kmh','MSE_LMMSE','MSE_CNN','InferredSpeed_kmh'}), ...
 fullfile(folder,'v2_tx_validation.csv'));
model.Signature=struct('Users',4,'UERx',4,'RFChains',8,'Qm',9,'M',512,'CSIWidth',24, ...
 'BeamWidth',cfg.BeamWidth,'CPFraction',cfg.CPFraction,'PhaseBits',4);
save(fullfile(folder,'v2_tx.mat'),'model','-v7.3');
fprintf('V2 transmitter learned blend %.6f, validation saved\n',model.BlendTX);
end
function [X,E,T,labels]=examples(records,model,r)
G=size(records{1}.Samples,1)/24;N=numel(records)*G;
X=zeros(6,4,256,N,'single');E=zeros(6,12,256,N,5,'single');
T=zeros(6,12,256,N,'single');labels=zeros(5,N,'single');
for index=1:numel(records)
 item=records{index};selected=(index-1)*G+(1:G);
 [x,e,t]=v2GateInputs(item,model,r);X(:,:,:,selected)=x;E(:,:,:,selected,:)=e;T(:,:,:,selected)=t;
 labels(find(model.Speeds==item.Speed),selected)=1;
end
end
