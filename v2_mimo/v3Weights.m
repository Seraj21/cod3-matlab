function [weights,diagnostic]=v3Weights(h,uncertainty,scheme,scene,r)
% A slot-coherent option keeps effective pilots and payload under one beam.
% The beam uses predicted CSI only; temporal disagreement adds design uncertainty.
if isfield(scene.Config,'PrecoderTimeMode') && string(scene.Config.PrecoderTimeMode)=="slot"
 times=scene.Config.DownlinkSymbols;
 center=mean(h(:,times,:,:),2);
 spread=mean(abs(h(:,times,:,:)-center).^2,2);
 uncertainty(:,times,:,:)=repmat(mean(uncertainty(:,times,:,:),2)+spread,1,numel(times),1,1);
 h(:,times,:,:)=repmat(center,1,numel(times),1,1);
end
[baseline,diagnostic]=v2Weights(h,scheme,scene,r);
if scheme=="PZF"
 [robust,d]=v2RobustPhaseWeights(h,uncertainty,scene,r,baseline);
 [weights,g]=v2PhaseGauge(robust,h,scene,r,"quantized");
 diagnostic.MaximumPowerError=max(d.MaximumPowerError,g.TotalPowerError);
 diagnostic.PhaseReference="quantized desired-channel reference";
else,weights=baseline;end
end
