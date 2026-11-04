function scene = muSounding(cfg,r,seed)
% Four orthogonal SRS-like comb pilots, a guard, then a seven-symbol DL.
cfg.SampleRate = r.OFDM.SampleRate;
front = muFrontEnd(cfg.Antennas,cfg.RFChains,cfg.VerticalCorrelation,cfg.HorizontalCorrelation);
channel = muChannel(cfg,front,cfg.SampleRate,seed);
channelInfo = info(channel);
tail = max(channelInfo.MaximumChannelDelay,ceil(12e-6*cfg.SampleRate));
userWave = nrOFDMModulate(r.Carrier,single(r.SRS),'Windowing',0);
N = size(userWave,1);
wave = complex(zeros(N,cfg.Users*cfg.UERx,'single'));
txPhase = cell(cfg.Users,1);
paGain = zeros(1,cfg.Users);
for u = 1:cfg.Users
    [distorted,txPhase{u},paGain(u)] = muRadio(userWave(:,u),cfg,seed+100000+u,true,0);
    columns = (u-1)*cfg.UERx+(1:cfg.UERx);
    wave(:,columns) = distorted*ones(1,cfg.UERx)/sqrt(cfg.UERx);
end
swapTransmitAndReceive(channel);
reset(channel);
[rx,Hul,trueOffset] = channel([wave;zeros(tail,size(wave,2),'single')],r.Carrier);
rx = rx.*front.ChannelGain.';
[rx,rxPhase] = muRadio(rx,cfg,seed+200000,false,0);
stream = RandStream('mt19937ar','Seed',seed+300000);
baseNoise = complex(randn(stream,size(rx)),randn(stream,size(rx)))/sqrt(2*double(r.OFDM.Nfft));
variance = 10^(-cfg.SoundingSNR/10);
noisy = double(rx)+sqrt(variance)*baseNoise;
maximum = ceil(12e-6*cfg.SampleRate);
offset = muTiming(r.Carrier,noisy,r.SRS,maximum);
frequency = muCFO(noisy,offset,r.OFDM,cfg.UplinkSymbols);
compensated = noisy.*exp(-1i*2*pi*frequency*(0:size(noisy,1)-1).'/cfg.SampleRate);
observed = nrOFDMDemodulate(r.Carrier,compensated(offset+(1:r.Samples),:));
samples = muPilotObservations(observed,r,"UL");
clean = double(rx).*exp(-1i*2*pi*frequency*(0:size(rx,1)-1).'/cfg.SampleRate);
clean = nrOFDMDemodulate(r.Carrier,clean(offset+(1:r.Samples),:));
cleanSamples = muPilotObservations(clean,r,"UL");
Hul = double(Hul(:,1:r.L,:,:)).*reshape(front.ChannelGain,1,1,cfg.RFChains,1);
truth = complex(zeros(r.K,r.L,cfg.RFChains,cfg.Users));
for u = 1:cfg.Users
    columns = (u-1)*cfg.UERx+(1:cfg.UERx);
    h = sum(Hul(:,:,:,columns),4)/sqrt(cfg.UERx)*paGain(u);
    truth(:,:,:,u) = muRFReference(h,txPhase{u},rxPhase,frequency,offset,trueOffset,r);
end
swapTransmitAndReceive(channel);
reset(channel);
scene = struct('Config',cfg,'Front',front,'Channel',channel,'Tail',tail, ...
    'TrueOffset',double(trueOffset),'UplinkOffset',offset,'UplinkCFO',frequency, ...
    'ULSamples',samples,'ULCleanSamples',cleanSamples,'ULTruth',truth,'ULNoise',muGuardNoise(noisy,r,tail), ...
    'HDownlink',permute(Hul,[1 2 4 3]),'Seed',seed);
noiseStream = RandStream('mt19937ar','Seed',seed+400000);
scene.DataNoise = complex(randn(noiseStream,r.Samples+tail,cfg.Users*cfg.UERx), ...
    randn(noiseStream,r.Samples+tail,cfg.Users*cfg.UERx))/sqrt(2*double(r.OFDM.Nfft));
end
