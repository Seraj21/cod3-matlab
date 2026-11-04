function [X,E,T,state,meta]=v2GateInputs(item,model,r)
state=v2CEBase(item.Samples,item.Noise,r.UplinkLayout,model.PriorsTX);
X=v2GateFeatures(item.Samples,item.Noise,r.UplinkLayout);
meta=struct('K',size(item.Samples,1),'L',12,'R',8,'U',4,'W',24, ...
 'G',size(item.Samples,1)/24,'GroupSize',8,'Blocks',1,'Coefficients',32);
E=zeros(6,12,256,meta.G,5,'single');
for k=1:5
 expert=v2CEBase(item.Samples,item.Noise,r.UplinkLayout,model.ExpertsTX{k});
 E(:,:,:,:,k)=muCNNTargets(expert.Wiener,state.Scale,meta);
end
T=[];if ~isempty(item.Truth),T=muCNNTargets(double(item.Truth),state.Scale,meta);end
end
