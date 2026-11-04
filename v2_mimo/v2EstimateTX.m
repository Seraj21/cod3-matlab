function [h,probabilities]=v2EstimateTX(samples,noise,model,r,method)
item=struct('Samples',samples,'Noise',noise,'Truth',[]);
if method~="CNN"
 state=v2CEBase(samples,noise,r.UplinkLayout,model.PriorsTX);
 h=state.Wiener;if method=="LS",h=state.LS;end
 probabilities=[];return
end
[X,E,~,state,meta]=v2GateInputs(item,model,r);
probabilities=double(extractdata(predict(model.NetworkTX,dlarray(X,'SSCB'))));
weights=reshape(probabilities.',1,1,1,meta.G,5);
packed=sum(double(E).*weights,5);
estimate=muCNNRestore(packed,meta,state.Scale);
h=state.Wiener+model.BlendTX*(estimate-state.Wiener);
end
