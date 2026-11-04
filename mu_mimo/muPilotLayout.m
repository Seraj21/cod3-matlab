function p = muPilotLayout(W,L,freq,times,noiseGain,label)
% Sampling and LS interpolation operators on one precoding resource group.
p = struct('W',W,'L',L,'Freq',freq(:),'Times',times(:),'NoiseGain',noiseGain,'Label',string(label));
p.Mask = false(W,L);
p.Mask(freq,times) = true;
p.Observed = find(p.Mask);
p.Count = numel(p.Observed);
f = interp1(double(freq(:)),eye(numel(freq)),min(max((1:W).',min(freq)),max(freq)),'linear');
t = interp1(double(times(:)),eye(numel(times)),min(max((1:L).',min(times)),max(times)),'linear');
p.Linear = kron(t,f).';
p.TimeCoordinate = repmat(linspace(-1,1,L),W,1);
p.FrequencyCoordinate = repmat(linspace(-1,1,W).',1,L);
end
