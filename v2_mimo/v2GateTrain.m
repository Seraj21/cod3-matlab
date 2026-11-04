function [net,history]=v2GateTrain(X,E,T,labels,V,VE,VT,Vlabels,epochs,folder)
rng(720001,'twister');net=v2GateNetwork(8,4);batchSize=64;
average=[];averageSquare=[];iteration=0;best=Inf;bestNet=net;stale=0;
lossFunction=dlaccelerate(@v2GateLoss);history=zeros(epochs,5);
for epoch=1:epochs
 order=randperm(size(X,4));trainLoss=0;batches=0;rate=1e-3*.5^floor((epoch-1)/10);
 for first=1:batchSize:numel(order)
  selected=order(first:min(first+batchSize-1,numel(order)));
  iteration=iteration+1;
  [loss,gradient]=dlfeval(lossFunction,net,dlarray(X(:,:,:,selected),'SSCB'), ...
    dlarray(E(:,:,:,selected,:)),dlarray(T(:,:,:,selected)),dlarray(labels(:,selected)));
  normSquared=sum(cellfun(@(g)sum(abs(extractdata(g)).^2,'all'),gradient.Value));
  factor=min(1,1/sqrt(max(normSquared,eps)));
  for j=1:height(gradient)
   gradient.Value{j}=gradient.Value{j}*factor+1e-5*net.Learnables.Value{j};
  end
  [net,average,averageSquare]=adamupdate(net,gradient,average,averageSquare,iteration,rate);
  trainLoss=trainLoss+double(extractdata(loss));batches=batches+1;
 end
 [validation,accuracy]=validate(net,V,VE,VT,Vlabels,batchSize);
 history(epoch,:)=[epoch trainLoss/batches validation accuracy rate];
 fprintf('Gate epoch %d: training %.6f, validation NMSE %.6f, speed accuracy %.3f\n', ...
     epoch,trainLoss/batches,validation,accuracy);
 if validation<best-1e-6
  best=validation;bestNet=net;stale=0;
 else,stale=stale+1;end
 if stale>=18,history=history(1:epoch,:);break;end
end
net=bestNet;
writetable(array2table(history,'VariableNames',{'Epoch','TrainingLoss','ValidationNMSE','SpeedAccuracy','LearningRate'}), ...
 fullfile(folder,'v2_gate_learning.csv'));
end
function [mse,accuracy]=validate(net,X,E,T,labels,batchSize)
total=0;elements=0;right=0;
time=reshape(single([.1 .1 .1 .1 .1 1 1 1 1 1 1 1]),1,12,1,1);
for first=1:batchSize:size(X,4)
 selected=first:min(first+batchSize-1,size(X,4));
 p=double(extractdata(predict(net,dlarray(X(:,:,:,selected),'SSCB'))));
 prediction=sum(double(E(:,:,:,selected,:)).*reshape(p.',1,1,1,numel(selected),5),5);
 error=(prediction-double(T(:,:,:,selected))).^2.*time;
 total=total+sum(error,'all');elements=elements+numel(error)*mean(time,'all');
 [~,guess]=max(p,[],1);[~,truth]=max(labels(:,selected),[],1);right=right+nnz(guess==truth);
end
mse=total/elements;accuracy=right/size(X,4);
end
