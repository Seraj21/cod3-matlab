function [loss,gradients]=v2GateLoss(net,X,E,T,label)
probabilities=stripdims(forward(net,X));B=size(T,4);
weights=reshape(probabilities.',1,1,1,B,5);
prediction=sum(E.*weights,5);
time=reshape(single([.1 .1 .1 .1 .1 1 1 1 1 1 1 1]),1,12,1,1);
fit=mean((prediction-T).^2.*time,'all')/mean(time,'all');
classification=-mean(sum(label.*log(max(probabilities,1e-6)),1));
loss=fit+.02*classification;
gradients=dlgradient(loss,net.Learnables);
end
