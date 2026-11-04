function model = muTrainNetworks(cfg,r,data,folder,epochs)
model = struct('Version',4,'PriorsTX',data.PriorsTX,'PriorsRX',data.PriorsRX, ...
    'BlendTX',1,'BlendRX',1,'BlendMultiplier',1,'TrainingFrames',numel(data.TrainTX), ...
    'ValidationFrames',numel(data.ValidTX));
model.Signature = struct('Users',cfg.Users,'UERx',cfg.UERx,'RFChains',cfg.RFChains, ...
    'NumRB',cfg.NumRB,'M',cfg.M,'Qm',cfg.Qm,'PhaseBits',cfg.PhaseBits);
for link = ["TX" "RX"]
    if link=="TX"
        training = data.TrainTX;validation=data.ValidTX;layouts=r.UplinkLayout;
        priors=model.PriorsTX;groupSize=cfg.RFChains;
    else
        training = data.TrainRX;validation=data.ValidRX;layouts=r.DownlinkLayout;
        priors=model.PriorsRX;groupSize=cfg.UERx;
    end
    [X,T] = examples(training,layouts,priors,r,groupSize);
    [V,Y] = examples(validation,layouts,priors,r,groupSize);
    rng(710001+(link=="RX"),'twister');
    C = groupSize*cfg.Users;
    net = muCNNArchitecture(size(X,3),C);
    opts = trainingOptions('adam','MaxEpochs',epochs,'MiniBatchSize',32, ...
        'InitialLearnRate',1e-3,'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',15,'LearnRateDropFactor',.4,'L2Regularization',1e-4, ...
        'GradientThreshold',1,'Shuffle','every-epoch','ExecutionEnvironment','cpu', ...
        'ValidationData',{V,Y},'ValidationFrequency',max(1,ceil(size(X,4)/32)), ...
        'ValidationPatience',10,'OutputNetwork','best-validation','Verbose',true, ...
        'VerboseFrequency',max(1,ceil(size(X,4)/32)));
    fprintf('Training joint %s CNN on %d MIMO patches, validating on %d patches\n',link,size(X,4),size(V,4));
    loss = @(predictions,targets)muTrainingLoss(predictions,targets,link);
    [net,info] = trainnet(X,T,net,loss,opts);
    try
        writetable(info.TrainingHistory,fullfile(folder,"learning_"+lower(link)+".csv"));
    catch
        save(fullfile(folder,"learning_"+lower(link)+".mat"),'info');
    end
    [priors,blend,summary] = calibrate(net,validation,layouts,priors,r,groupSize,link);
    writetable(summary,fullfile(folder,"validation_channel_"+lower(link)+".csv"));
    if link=="TX"
        model.NetworkTX=net;model.PriorsTX=priors;model.BlendTX=blend;
    else
        model.NetworkRX=net;model.PriorsRX=priors;model.BlendRX=blend;
    end
    clear X T V Y net
end
end
function [X,Y] = examples(records,layouts,priors,r,groupSize)
features = cell(numel(records),1);
targets = cell(numel(records),1);
for i = 1:numel(records)
    item=records{i};
    [~,~,state] = muCE(item.Samples,item.Noise,layouts,priors,r,"LS");
    [features{i},meta] = muCNNFeatures(state,layouts,r,groupSize);
    targets{i} = muCNNTargets(double(item.Truth)-state.Wiener,state.Scale,meta);
end
X=cat(4,features{:});Y=cat(4,targets{:});
end
function [priors,blend,summary] = calibrate(net,records,layouts,priors,r,groupSize,link)
keys = string(fieldnames(priors));
sums = zeros(numel(keys),5);
numerator=0;denominator=0;
rows = zeros(numel(records),4);
for i = 1:numel(records)
 item=records{i};
 [~,~,state]=muCE(item.Samples,item.Noise,layouts,priors,r,"LS");
 [X,meta]=muCNNFeatures(state,layouts,r,groupSize);
 z=double(extractdata(predict(net,dlarray(X,'SSCB'))));
 correction=muCNNRestore(z,meta,state.Scale);
 truth=double(item.Truth);
 times=6:12;
 error=truth(:,times,:,:)-state.Wiener(:,times,:,:);
 update=correction(:,times,:,:);
 numerator=numerator+real(sum(conj(update).*error,'all'));
 denominator=denominator+sum(abs(update).^2,'all');
 for u=1:numel(layouts)
  k=find(keys==layouts{u}.Label);
  eLS=truth(:,times,:,u)-state.LS(:,times,:,u);
  eW=truth(:,times,:,u)-state.Wiener(:,times,:,u);
  eC=eW-correction(:,times,:,u);
  vLS=state.VarianceLS(:,times,:,u);vW=state.VarianceWiener(:,times,:,u);
  sums(k,:)=sums(k,:)+[sum(abs(eLS).^2,'all') sum(abs(eW).^2,'all') ...
      sum(abs(eC).^2,'all') sum(vLS,'all') sum(vW,'all')];
 end
 rows(i,:)=[item.Seed mean(abs(truth(:,times,:,:)-state.LS(:,times,:,:)).^2,'all') ...
     mean(abs(error).^2,'all') mean(abs(error-update).^2,'all')];
end
blend=max(0,min(1,numerator/max(denominator,eps)));
for k=1:numel(keys)
 m=priors.(keys(k));
 m.VarianceFactor=struct('LS',max(1e-4,sums(k,1)/max(sums(k,4),eps)), ...
     'LMMSE',max(1e-4,sums(k,2)/max(sums(k,5),eps)), ...
     'CNN',max(1e-4,sums(k,3)/max(sums(k,5),eps)));
 priors.(keys(k))=m;
end
summary=array2table(rows,'VariableNames',{'ChannelSeed','MSE_LS','MSE_LMMSE','MSE_CNN_FullResidual'});
fprintf('%s validation-trained residual blend %.5f\n',link,blend);
end
