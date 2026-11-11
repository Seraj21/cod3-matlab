function v2_probe_physics(scenario,antennas,numRB,frames,scheme)
if nargin>0
 if nargin<2,antennas=32;end
 if nargin<3,numRB=264;end
 if nargin<4,frames=2;end
 if nargin<5,scheme="ZF";end
 currentDiagnostic(string(scenario),antennas,numRB,frames,string(scheme));
 return
end
% Controlled validation: retain TDL-C/512-QAM/LDPC R=1/2 and RF impairments.
addpath('mu_mimo');
cfg=muConfig(64);r=muResources(cfg);
rows=cell(0,12);
for packetIndex=1:2
 scene=muSounding(cfg,r,6100000+packetIndex);
 [grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 for width=[24 12 4 1]
  for align=[true false]
   weights=makeWeights(scene.ULTruth,width,align,scene,r);
   [packet,gains,times,filters]=physicalWaveform(scene,r,grid,weights);
   for fraction=[.5 .75 1]
    H=nrPerfectChannelEstimate(r.Carrier,gains,filters,packet.TrueOffset,times, ...
        'CyclicPrefixFraction',fraction);
    H=double(H(:,1:r.L,:,:)).*reshape(scene.Front.ChannelGain,1,1,1,8);
    H=permute(pagemtimes(permute(H,[3 4 1 2]), ...
        double(weights).*reshape(packet.PAGain,8,1,1,1)),[3 4 1 2]);
    for snr=[10 20]
     input=receive(packet,H,scene,r,snr,fraction);
     predicted=sum(input.Truth.*reshape(grid,r.K,r.L,1,4),4);
     mask=repmat(r.DataMask,1,1,16);
     evm=sum(abs(input.Observed(mask)-predicted(mask)).^2)/sum(abs(predicted(mask)).^2);
     for combiner=["MMSE" "SRS beam"]
      [observed,h,noise,decodeConfig]=combine(input,scene.Config,r,combiner);
      [errors,failed,details]=muDecode(struct('Observed',observed,'Noise',noise), ...
          h,zeros(size(h)),payload,decodeConfig,r);
      correct=nnz(errors==0 & ~failed);
      rows(end+1,:)={packetIndex,width,align,fraction,snr,combiner, ...
          sum(errors),numel(payload),correct, ...
          correct*r.PayloadBits/r.FrameDuration/1e6,evm,mean(details.MeanEffectiveVariance)};
     end
    end
   end
  end
 end
 fprintf('Completed physical validation frame %d/2\n',packetIndex);
end
T=cell2table(rows,'VariableNames',{'Frame','PrecodingSubcarriers','AlignPrecoderPhase', ...
 'CyclicPrefixFraction','SNR_dB','Combiner','PayloadBitErrors','PayloadBits', ...
 'CorrectBlocks','Goodput_Mbps','WaveformRelativeMSE','LLRVariance'});
writetable(T,'v2_physics.csv');
fprintf('V2_PHYSICS_BEGIN\n%s\nV2_PHYSICS_END\n',jsonencode(table2struct(T)));
end
function weights=makeWeights(h,width,align,scene,r)
weights=complex(zeros(8,4,r.K,r.L,'single'));
G=r.K/width;previous=cell(G,4);
for l=6:12
 for g=1:G
  band=(g-1)*width+(1:width);
  H=reshape(mean(h(band,l,:,:),1),8,4).';
  B=muPrecoder(double(H),"ZF",scene.Front,4,2);
  for u=1:4
   if align
    prior=previous{g,u};
    if isempty(prior) && g>1,prior=previous{g-1,u};end
    if ~isempty(prior),B(:,u)=B(:,u)*exp(-1i*angle(prior'*B(:,u)));end
    previous{g,u}=B(:,u);
   end
  end
  weights(:,:,band,l)=repmat(single(B),1,1,width);
 end
end
end
function [packet,pg,st,pf]=physicalWaveform(scene,r,grid,weights)
cfg=scene.Config;
x=sum(permute(weights,[3 4 1 2]).*reshape(single(grid),r.K,r.L,1,4),4);
wave=nrOFDMModulate(r.Carrier,x,'Windowing',0);
[distorted,txPhase,paGain]=muRadio(wave,cfg,muSeed(scene.Seed,15),true,0);
channel=muChannel(cfg,scene.Front,cfg.SampleRate,scene.Seed);
channel.ChannelResponseOutput='path-gains';
[rx,pg,st]=channel([distorted.*scene.Front.ChannelGain.';zeros(scene.Tail,8,'single')]);
pf=getPathFilters(channel);offset=nrPerfectTimingEstimate(pg,pf);
rxPhase=cell(4,1);
for u=1:4
 columns=(u-1)*4+(1:4);
 [rx(:,columns),rxPhase{u}]=muRadio(rx(:,columns),cfg,muSeed(scene.Seed,16,u),false,0);
end
packet=struct('Waveform',double(rx),'TxPhase',txPhase,'RxPhase',{rxPhase}, ...
 'TrueOffset',offset,'PAGain',paGain);
end
function input=receive(packet,H,scene,r,snr,fraction)
cfg=scene.Config;y=packet.Waveform+10^(-snr/20)*scene.DataNoise;
observed=complex(zeros(r.K,r.L,16));truth=H;
for u=1:4
 columns=(u-1)*4+(1:4);
 practical=muTiming(r.Carrier,y(:,columns),r.DMRS,ceil(12e-6*cfg.SampleRate));
 f=muCFO(y(:,columns),practical,r.OFDM,6:12);
 z=y(:,columns).*exp(-1i*2*pi*f*(0:size(y,1)-1).'/cfg.SampleRate);
 observed(:,:,columns)=nrOFDMDemodulate(r.Carrier, ...
     z(packet.TrueOffset+(1:r.Samples),:),'CyclicPrefixFraction',fraction);
 start=[0 cumsum(double(r.OFDM.SymbolLengths(1:r.L-1)))];
 phase=complex(zeros(1,r.L));
 for l=1:r.L
  tx=start(l)+floor(fraction*double(r.OFDM.CyclicPrefixLengths(l)))+(1:r.OFDM.Nfft);
  rx=tx+packet.TrueOffset;
  rot=exp(-1i*2*pi*f*(rx-1)/cfg.SampleRate);
  phase(l)=mean(packet.TxPhase(tx).*packet.RxPhase{u}(rx).*rot.');
 end
 truth(:,:,columns,:)=H(:,:,columns,:).*reshape(phase,1,r.L,1,1);
end
input=struct('Observed',observed,'Truth',truth,'Noise', ...
 repmat(muGuardNoise(y,r,scene.Tail),1,16));
end
function [y,h,n,cfg]=combine(input,cfg,r,kind)
y=input.Observed;h=input.Truth;n=input.Noise;
if kind=="SRS beam"
 y=reshape(sum(reshape(y,r.K,r.L,4,4),3)/2,r.K,r.L,4);
 h=reshape(sum(reshape(h,r.K,r.L,4,4,4),3)/2,r.K,r.L,4,4);
 n=n(1:4:end);cfg.UERx=1;
end
end

function currentDiagnostic(scenario,antennas,numRB,frames,scheme)
% Causal controls use held-out paired seeds. Oracle variants are diagnostics only.
addpath(pwd,'mu_mimo','v2_mimo');
loaded=load('v3_refined/v3_model.mat','model');model=loaded.model;
cfg=v3Config(antennas,numRB,"stressed");perfectTX=false;perfectTiming=false;
switch scenario
 case "baseline"
 case "delay300",cfg.DelaySpread=300e-9;
 case "delay30",cfg.DelaySpread=30e-9;
 case "static",cfg.Speed=0;
 case "sounding35",cfg.SoundingSNR=35;
 case "quietRF",cfg=quietRadio(cfg);
 case "perfectTX",perfectTX=true;
 case "perfectTiming",perfectTiming=true;
 case "fftEnd",cfg.CPFraction=1;
 case "operatingPoint"
  cfg.DelaySpread=300e-9;cfg.Speed=30;cfg.SoundingSNR=30;
 case "idealWaveform"
  cfg=quietRadio(cfg);cfg.DelaySpread=30e-9;cfg.Speed=0;
  perfectTX=true;perfectTiming=true;
 otherwise,error('Unknown diagnostic scenario: %s',scenario);
end
% Measure one receive combiner consistently, including its pre-LDPC metrics.
cfg.DecoderDual=false;r=v3Resources(cfg);rows=cell(0,22);
for frame=1:frames
 scene=v2Sounding(cfg,r,96000000+frame);
 [grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
 csit=tx.Wiener;uncertainty=tx.VarianceWiener;
 if perfectTX,csit=scene.ULTruth;uncertainty=zeros(size(csit));end
 [weights,power]=v3Weights(csit,uncertainty,scheme,scene,r);
 assert(power.MaximumPowerError<1e-6);
 packet=v2Waveform(scene,r,grid,weights,true);
 for snr=40
  input=v3Receive(packet,scene,r,snr,perfectTiming,false);
  prediction=sum(input.Truth.*reshape(grid,r.K,r.L,1,cfg.Users),4);
  mask=repmat(r.DataMask,1,1,cfg.Users*cfg.UERx);
  waveformMSE=sum(abs(input.Observed(mask)-prediction(mask)).^2)/max(sum(abs(prediction(mask)).^2),eps);
  for method=["LMMSE" "Perfect"]
   [h,v,observed]=v3EstimateRX(input,model.Receivers.(scheme),r,method);
   local=input;local.Observed=observed;
   [e,f,detail,primary]=v2Decode(local,h,v,payload,cfg,r);
   [decoded,f]=v4Retry(primary,f,cfg,r,"APP50");e=sum(decoded~=payload,1);
   for u=1:cfg.Users
    coded=r.Codec.encode(payload(:,u));hard=primary.LLR(:,u)<0;
    symbols=grid(:,:,u);symbols=symbols(r.DataIndices);
    mse=mean(abs(primary.Symbols(:,u)-symbols).^2)/mean(abs(symbols).^2);
    good=~f(u) && e(u)==0;
    rows(end+1,:)={char(scenario),antennas,numRB,frame,scene.Seed,char(scheme),char(method),snr,u, ...
     cfg.DelaySpread*1e9,cfg.Speed,cfg.SoundingSNR,r.PayloadBits,e(u),double(f(u)), ...
     double(good)*r.PayloadBits/r.FrameDuration/1e6,mean(hard~=coded),100*sqrt(mse), ...
     -10*log10(max(mean(primary.Variance(:,u)),eps)),waveformMSE, ...
     input.TimingOffset(u)-packet.TrueOffset,input.CFO(u)-cfg.FrequencyOffset}; %#ok<AGROW>
   end
  end
 end
 T=cell2table(rows,'VariableNames',{'Scenario','BS','NumRB','Frame','ChannelSeed','Precoder','Estimator', ...
  'SNR_dB','UE','DelaySpread_ns','Speed_kmh','SoundingSNR_dB','PayloadBits','Errors','CRCFailed', ...
  'Goodput_Mbps','PrimaryPreLDPC_BER','PrimaryEVM_percent','PrimaryEstimatedSINR_dB', ...
  'WaveformRelativeMSE','TimingError_samples','CFOResidual_Hz'});
 if ~isfolder('physics_diagnostic'),mkdir('physics_diagnostic');end
 writetable(T,fullfile('physics_diagnostic',scenario+"_"+scheme+"_BS"+antennas+".csv"));
 fprintf('PHYSICS_DIAGNOSTIC_FRAME %s %s BS%d frame%d: %s\n', ...
  scenario,scheme,antennas,frame,jsonencode(table2struct(T(T.Frame==frame,:))));
end
if scenario=="idealWaveform"
 selected=T.Estimator=="Perfect";
 assert(all(T.Errors(selected)==0 & ~T.CRCFailed(selected)), ...
  'Clean waveform with perfect CSI must decode at 40 dB before any performance claim');
end
end
function cfg=quietRadio(cfg)
cfg.Quiet=true;cfg.PhaseNoiseLevel=[-300 -300 -300];
cfg.InputBackoff=80;cfg.AMPMDegrees=0;cfg.FrequencyOffset=0;
end
