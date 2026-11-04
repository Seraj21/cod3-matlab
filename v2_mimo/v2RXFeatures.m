function [X,T,meta]=v2RXFeatures(state,truth,layouts,r)
[X,meta]=muCNNFeatures(state,layouts,r,4);
scale=reshape(repelem(state.Scale,meta.W,1,1),meta.K,1,meta.R,meta.U);
scale=repmat(scale,1,meta.L,1,1);
logvar=log(max(1e-6,state.RawVarianceWiener./scale.^2));
packed=v2ParameterPatches(logvar,meta);
X=cat(3,X,packed);
T=[];
if ~isempty(truth)
 target=muCNNTargets(double(truth)-state.Wiener,state.Scale,meta);
 T=cat(3,target,packed);
end
end
