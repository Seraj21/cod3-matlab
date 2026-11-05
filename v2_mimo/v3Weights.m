function [weights,diagnostic]=v3Weights(h,uncertainty,scheme,scene,r)
[baseline,diagnostic]=v2Weights(h,scheme,scene,r);
if scheme=="PZF"
 [robust,d]=v2RobustPhaseWeights(h,uncertainty,scene,r,baseline);
 [weights,g]=v2PhaseGauge(robust,h,scene,r,"quantized");
 diagnostic.MaximumPowerError=max(d.MaximumPowerError,g.TotalPowerError);
 diagnostic.PhaseReference="quantized desired-channel reference";
else,weights=baseline;end
end
