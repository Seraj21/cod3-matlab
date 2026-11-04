function input = muReceive(packet,scene,r,snrDb,perfectTiming,includeClean)
arguments
    packet
    scene
    r
    snrDb (1,1) double
    perfectTiming (1,1) logical = false
    includeClean (1,1) logical = false
end
cfg = scene.Config;
variance = 10^(-snrDb/10);
noisy = packet.Waveform+sqrt(variance)*scene.DataNoise;
observed = complex(zeros(r.K,r.L,cfg.Users*cfg.UERx));
cleanObserved = observed;
truth = complex(zeros(r.K,r.L,cfg.Users*cfg.UERx,cfg.Users));
noise = zeros(1,cfg.Users*cfg.UERx);
timing = zeros(1,cfg.Users);
frequency = zeros(1,cfg.Users);
for u = 1:cfg.Users
    rows = (u-1)*cfg.UERx+(1:cfg.UERx);
    y = noisy(:,rows);
    practicalOffset = muTiming(r.Carrier,y,r.DMRS,ceil(12e-6*cfg.SampleRate));
    f = muCFO(y,practicalOffset,r.OFDM,cfg.DownlinkSymbols);
    offset = practicalOffset;
    if perfectTiming,offset=packet.TrueOffset;end
    rotation = exp(-1i*2*pi*f*(0:size(y,1)-1).'/cfg.SampleRate);
    y = y.*rotation;
    observed(:,:,rows) = nrOFDMDemodulate(r.Carrier,y(offset+(1:r.Samples),:));
    if includeClean
        c = packet.Waveform(:,rows).*rotation;
        cleanObserved(:,:,rows) = nrOFDMDemodulate(r.Carrier,c(offset+(1:r.Samples),:));
    end
    truth(:,:,rows,:) = muRFReference(packet.Propagation(:,:,rows,:),packet.TxPhase, ...
        packet.RxPhase{u},f,offset,packet.TrueOffset,r);
    noise(rows) = muGuardNoise(noisy(:,rows),r,scene.Tail);
    timing(u) = abs(offset-packet.TrueOffset);
    frequency(u) = cfg.FrequencyOffset-f;
end
input = struct('Observed',observed,'Samples',muPilotObservations(observed,r,"DL"), ...
    'Truth',truth,'Noise',noise,'TimingError',timing,'CFOResidual',frequency);
if includeClean,input.CleanSamples=muPilotObservations(cleanObserved,r,"DL");end
end
