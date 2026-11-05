function input=v3Receive(packet,scene,r,snr,perfectTiming,includeClean)
if nargin<6,includeClean=false;end
input=v2Receive(packet,scene,r,snr,perfectTiming,includeClean);
input.Samples=v3PilotObservations(input.Observed,r);
if includeClean,input.CleanSamples=v3PilotObservations(input.CleanObserved,r);end
end
