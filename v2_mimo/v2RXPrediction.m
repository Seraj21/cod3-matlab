function [correction,rawVariance]=v2RXPrediction(prediction,state,meta)
C=4*meta.Coefficients;
correction=muCNNRestore(prediction(:,:,1:2*C,:),meta,state.Scale);
ratio=v2ParameterRestore(prediction(:,:,2*C+(1:C),:),meta);
rawVariance=state.RawVarianceWiener.*exp(max(-5,min(5,double(ratio))));
end
