function input=v2Receive(packet,scene,r,snr,perfectTiming,includeClean)
if nargin<5,perfectTiming=false;end
if nargin<6,includeClean=false;end
cfg=scene.Config;
y=packet.Waveform+10^(-snr/20)*scene.DataNoise;
% One correlation call returns all receive-antenna magnitudes exactly.
[~,magnitude]=nrTimingEstimate(r.Carrier,y,r.DMRS);
limit=min(size(magnitude,1),ceil(12e-6*cfg.SampleRate)+1);
observed=complex(zeros(r.K,r.L,cfg.Users*cfg.UERx));clean=observed;
truth=[];
if ~isempty(packet.Propagation),truth=packet.Propagation;end
offsets=zeros(1,cfg.Users);frequency=offsets;
noise=muGuardNoise(y,r,scene.Tail)*ones(1,cfg.Users*cfg.UERx);
start=[0 cumsum(double(r.OFDM.SymbolLengths(1:r.L-1)))];
for u=1:cfg.Users
 columns=(u-1)*cfg.UERx+(1:cfg.UERx);
 energy=sum(abs(magnitude(1:limit,columns)).^2,2);
 [~,peak]=max(energy);practical=peak-1;
 f=muCFO(y(:,columns),practical,r.OFDM,cfg.DownlinkSymbols);
 offset=practical;if perfectTiming,offset=packet.TrueOffset;end
 rotation=exp(-1i*2*pi*f*(0:size(y,1)-1).'/cfg.SampleRate);
 z=y(:,columns).*rotation;
 observed(:,:,columns)=nrOFDMDemodulate(r.Carrier,z(offset+(1:r.Samples),:), ...
     'CyclicPrefixFraction',cfg.CPFraction);
 if includeClean
  z=packet.Waveform(:,columns).*rotation;
  clean(:,:,columns)=nrOFDMDemodulate(r.Carrier,z(offset+(1:r.Samples),:), ...
      'CyclicPrefixFraction',cfg.CPFraction);
 end
 if ~isempty(truth)
  phase=complex(zeros(1,r.L));
  for l=1:r.L
   tx=start(l)+floor(cfg.CPFraction*double(r.OFDM.CyclicPrefixLengths(l)))+(1:r.OFDM.Nfft);
   rx=tx+packet.TrueOffset;
   phase(l)=mean(packet.TxPhase(tx).*packet.RxPhase{u}(rx).* ...
       exp(-1i*2*pi*f*(rx-1)/cfg.SampleRate).');
  end
  step=2*pi*(-r.K/2:r.K/2-1).'/r.OFDM.Nfft;
  truth(:,:,columns,:)=packet.Propagation(:,:,columns,:).*reshape(phase,1,r.L,1,1) ...
      .*exp(1i*step*(offset-packet.TrueOffset));
 end
 offsets(u)=offset;frequency(u)=f;
end
input=struct('Observed',observed,'Samples',muPilotObservations(observed,r,"DL"), ...
 'Noise',noise,'Truth',truth,'TimingOffset',offsets,'CFO',frequency);
if includeClean,input.CleanSamples=muPilotObservations(clean,r,"DL");end
end
