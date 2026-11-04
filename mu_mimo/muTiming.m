function offset = muTiming(carrier,waveform,reference,maximum)
[~,magnitude] = nrTimingEstimate(carrier,waveform,reference);
energy = sum(abs(magnitude(1:min(maximum+1,size(magnitude,1)),:)).^2,2);
[~,peak] = max(energy);
offset = peak-1;
end
