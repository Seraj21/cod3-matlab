function v2_probe_physics(scenario,antennas,numRB,frames,scheme)
if nargin>0
 if nargin<2,antennas=32;end
 if nargin<3,numRB=264;end
 if nargin<4,frames=2;end
 if nargin<5,scheme="ZF";end
 if string(scenario)=="firstPassJoint"
  firstPassJoint(antennas,numRB,frames,string(scheme));
 elseif string(scenario)=="harqRecovery"
  harqRecovery(antennas,numRB,frames,string(scheme));
 elseif string(scenario)=="receiverAudit"
  receiverDiagnostic(antennas,numRB,frames,string(scheme));
 elseif startsWith(string(scenario),"pilotBlock")
  pilotDiagnostic(str2double(extractAfter(string(scenario),"pilotBlock")),antennas,numRB,frames,string(scheme),"per-symbol");
 elseif startsWith(string(scenario),"pilotSlot")
  pilotDiagnostic(str2double(extractAfter(string(scenario),"pilotSlot")),antennas,numRB,frames,string(scheme),"slot");
 elseif startsWith(string(scenario),"pilotFDMA")
  pilotDiagnostic(str2double(extractAfter(string(scenario),"pilotFDMA")),antennas,numRB,frames,string(scheme),"slot",true);
 else
  currentDiagnostic(string(scenario),antennas,numRB,frames,string(scheme));
 end
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

function pilotDiagnostic(width,antennas,numRB,frames,scheme,timeMode,orthogonal)
if nargin<7,orthogonal=false;end
% Paired deployment audit: independently refit each changed pilot/precoder model.
addpath(pwd,'mu_mimo','v2_mimo');
loaded=load(fullfile('pilot_calibration','integrated_receiver',lower(scheme)+"_receiver.mat"),'receiver');
receiver=loaded.receiver;
assert(ismember(width,[1 4 12 24]));
cfg=v3Config(32,66,"mobile");cfg.BeamWidth=width;cfg.PrecoderTimeMode=char(timeMode);cfg.OrthogonalDMRS=orthogonal;r=v3Resources(cfg);
layouts=cell(1,8);
for u=1:4
 layouts{u}=r.DownlinkLayout{u};layouts{u}.Label="Own_"+layouts{u}.Label;
 layouts{u+4}=r.DownlinkLayout{u};layouts{u+4}.Label="Other_"+layouts{u+4}.Label;
end
stats=v2Prior("init",layouts,4);
for frame=1:24
 cfg=v3Config(32+32*mod(frame-1,2),66,"mobile");cfg.BeamWidth=width;cfg.PrecoderTimeMode=char(timeMode);cfg.OrthogonalDMRS=orthogonal;
 scene=v2Sounding(cfg,r,99000000+frame);
 tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,receiver.PriorsTX);
 weights=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
 [grid,~]=muPayload(cfg,r,muSeed(scene.Seed,70));
 packet=v2Waveform(scene,r,grid,weights,true);
 input=v3Receive(packet,scene,r,40,false,true);
 for u=1:4
  [clean,truth,layout]=v2RXCanonical(input,r,u,true);
  stats=v2Prior("accumulate",stats,clean,truth,layout);
 end
end
receiver.PriorsRX=v2Prior("finalize",stats);clear stats
cfg=v3Config(antennas,numRB,"mobile");cfg.BeamWidth=width;cfg.PrecoderTimeMode=char(timeMode);cfg.OrthogonalDMRS=orthogonal;r=v3Resources(cfg);
rows=cell(0,16);
for frame=1:frames
 scene=v2Sounding(cfg,r,98000000+frame);
 tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,receiver.PriorsTX);
 weights=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
 [grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 packet=v2Waveform(scene,r,grid,weights,true);
 input=v3Receive(packet,scene,r,40,false,false);
 [oracle,~,~]=v3EstimateRX(input,receiver,r,"Perfect");
 for method=["LS" "LMMSE" "Perfect"]
  [h,v,observed]=v3EstimateRX(input,receiver,r,method);local=input;local.Observed=observed;
  [~,failed,~,primary]=v2Decode(local,h,v,payload,cfg,r);
  [decoded,failed]=v4Retry(primary,failed,cfg,r,"APP50");errors=sum(decoded~=payload,1);
  for u=1:4
   symbols=grid(:,:,u);symbols=symbols(r.DataIndices);coded=r.Codec.encode(payload(:,u));
   evm=100*sqrt(mean(abs(primary.Symbols(:,u)-symbols).^2)/mean(abs(symbols).^2));
   branches=(u-1)*4+(1:4);mask=repmat(r.DataMask,1,1,4,4);
   delta=h(:,:,branches,:)-oracle(:,:,branches,:);target=oracle(:,:,branches,:);
   nmse=sum(abs(delta(mask)).^2)/max(sum(abs(target(mask)).^2),eps);
   good=~failed(u) && errors(u)==0;
   rows(end+1,:)={char(scheme),width,antennas,frame,u,char(method),r.PayloadBits,errors(u), ...
    double(failed(u)),double(good)*r.PayloadBits/r.FrameDuration/1e6, ...
    mean((primary.LLR(:,u)<0)~=coded),evm,nmse,mean(primary.Variance(:,u)),char(timeMode),orthogonal}; %#ok<AGROW>
  end
 end
end
T=cell2table(rows,'VariableNames',{'Precoder','BeamWidth','BS','Frame','UE','Estimator','PayloadBits', ...
 'Errors','CRCFailed','Goodput_Mbps','PreLDPC_BER','EVM_percent','ChannelNMSE','EffectiveVariance','PrecoderTimeMode','OrthogonalDMRS'});
if ~isfolder('physics_diagnostic'),mkdir('physics_diagnostic');end
writetable(T,fullfile('physics_diagnostic',"pilots_"+scheme+"_W"+width+"_"+timeMode+".csv"));
fprintf('PILOT_AUDIT_RESULT %s\n',jsonencode(table2struct(T)));
end

function receiverDiagnostic(antennas,numRB,frames,scheme)
% Frozen-receiver ablation. Oracle controls below are NOT deployable receivers.
% No fitting/training is performed on these diagnostic channels.
addpath(pwd,'mu_mimo','v2_mimo');
modelFile=fullfile('receiver_audit','integrated_receiver',lower(scheme)+"_receiver.mat");
loaded=load(modelFile,'receiver');receiver=loaded.receiver;
cfg=v3Config(antennas,numRB,"mobile");r=v3Resources(cfg);
assert(receiver.Scheme==scheme && receiver.TrainingCodeRate==cfg.Rate && receiver.ProjectionLoss);
for name=string(fieldnames(receiver.TrainingChannelConfig)).'
 assert(isequal(receiver.TrainingChannelConfig.(name),cfg.(name)), ...
  'Frozen receiver channel configuration mismatch: %s',name);
end
% Keep the same first-pass four-branch combiner for every ablation.
cfg.DecoderDual=false;
controls=["Baseline" "PTRSAmplitude" "OracleSymbolGain" "OracleVariance" "OracleGainVariance"];
% Accumulate scalar records in cells; fieldless struct([]) rejects indexed
% assignment of a record with fields in MATLAB.
rows={};symbolRows={};
for frame=1:frames
 scene=v2Sounding(cfg,r,99600000+frame);
 tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,receiver.PriorsTX);
 [weights,power]=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
 assert(power.MaximumPowerError<1e-6);
 [grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 packet=v2Waveform(scene,r,grid,weights,true);
 input=v3Receive(packet,scene,r,40,false,false);
 oracleChannel=input.Truth;input.Truth=[];
 for method=["LMMSE" "CNN" "Perfect"]
  local=input;
  if method=="Perfect",local.Truth=oracleChannel;end
  [h,v,observed]=v3EstimateRX(local,receiver,r,method);local.Observed=observed;
  [~,~,~,primary]=v2Decode(local,h,v,payload,cfg,r);
  for u=1:cfg.Users
   reference=grid(:,:,u);reference=reference(r.DataIndices);
   coded=r.Codec.encode(payload(:,u));original=primary.Symbols(:,u);
   baseVariance=primary.Variance(:,u);gain=ones(r.L,1);
   for l=unique(r.DataTimes).'
    mask=r.DataTimes==l;
    gain(l)=sum(original(mask).*conj(reference(mask)))/sum(abs(reference(mask)).^2);
    residual=original(mask)-reference(mask);
    symbolRows{end+1}=struct('Precoder',char(scheme),'BS',antennas,'Frame',frame, ...
     'ChannelSeed',scene.Seed,'Estimator',char(method),'UE',u,'Symbol',l, ...
     'PTRSAmplitude',primary.PTRSAmplitude(l,u),'OracleGainAmplitude',abs(gain(l)), ...
     'OracleGainPhase_deg',angle(gain(l))*180/pi,'MeasuredMSE',mean(abs(residual).^2), ...
     'EstimatedVariance',mean(baseVariance(mask)), ...
     'ScalarGainErrorFraction',sum(abs((gain(l)-1)*reference(mask)).^2)/max(sum(abs(residual).^2),eps)); %#ok<AGROW>
   end
   for control=controls
    z=original;effectiveVariance=baseVariance;divisor=ones(size(z));
    if control=="PTRSAmplitude"
     divisor=primary.PTRSAmplitude(r.DataTimes,u);
    elseif any(control==["OracleSymbolGain" "OracleGainVariance"])
     divisor=gain(r.DataTimes);
    end
    invalid=~isfinite(divisor) | abs(divisor)<1e-3;
    divisor(invalid)=1;
    z=z./divisor;effectiveVariance=effectiveVariance./abs(divisor).^2;
    if any(control==["OracleVariance" "OracleGainVariance"])
     % Truth-assisted diagnostic scale only; preserve relative per-tone weights.
     for l=unique(r.DataTimes).'
      mask=r.DataTimes==l;
      scale=mean(abs(z(mask)-reference(mask)).^2)/max(mean(effectiveVariance(mask)),1e-7);
      effectiveVariance(mask)=effectiveVariance(mask)*scale;
     end
    end
    effectiveVariance=max(effectiveVariance,1e-7);
    assert(all(isfinite(z)) && all(isfinite(effectiveVariance)));
    demapped=v4SymbolLLR(z,effectiveVariance,zeros(size(z)),cfg.M);
    restored=zeros(r.CodedBits,1);restored(r.BitOrder)=max(-50,min(50,demapped));
    % All controls use the identical exact-APP demapper and 50-iteration decoder.
    [decoded,failed]=r.Codec.decode(restored,50,'Layered belief propagation');
    errors=nnz(decoded~=payload(:,u));correct=~failed && errors==0;
    mse=mean(abs(z-reference).^2);oracleControl=startsWith(control,"Oracle");
    rows{end+1}=struct('Precoder',char(scheme),'BS',antennas,'NumRB',numRB, ...
     'Frame',frame,'ChannelSeed',scene.Seed,'SNR_dB',40,'Estimator',char(method),'UE',u, ...
     'Control',char(control),'UsesPayloadTruth',oracleControl, ...
     'UsesTrueChannel',method=="Perfect",'PayloadBits',r.PayloadBits,'Errors',errors, ...
     'CRCFailed',logical(failed),'CorrectBlock',correct, ...
     'Goodput_Mbps',double(correct)*r.PayloadBits/r.FrameDuration/1e6, ...
     'PreLDPC_BER',mean((restored<0)~=coded), ...
     'EVM_percent',100*sqrt(mse/mean(abs(reference).^2)), ...
     'MeasuredMSE',mse,'EstimatedVariance',mean(effectiveVariance), ...
     'MSEToVarianceRatio',mse/mean(effectiveVariance),'GainFallbackSymbols',nnz(invalid)); %#ok<AGROW>
   end
  end
 end
 if ~isfolder('physics_diagnostic'),mkdir('physics_diagnostic');end
 T=struct2table(vertcat(rows{:}));S=struct2table(vertcat(symbolRows{:}));
 prefix=fullfile('physics_diagnostic',"receiver_"+scheme+"_BS"+antennas);
 writetable(T,prefix+".csv");writetable(S,prefix+"_symbols.csv");
 settings=struct('Config',cfg,'SourceRun',43193721852,'ModelFile',modelFile, ...
  'Decoder','APP50, four-branch first pass only','SeedOffset',99600000, ...
  'CompletedFrames',frame,'FrameDuration',r.FrameDuration, ...
  'Warning','Oracle controls use payload/channel truth and are diagnostics only.');
 save(prefix+"_settings.mat",'settings');
 for method=["LMMSE" "CNN" "Perfect"]
  for control=controls
   t=T(string(T.Estimator)==method & string(T.Control)==control,:);
   summary=struct('Precoder',scheme,'BS',antennas,'Estimator',method,'Control',control, ...
    'DiagnosticOnly',true,'Frames',frame,'SNR_dB',40,'BER',sum(t.Errors)/sum(t.PayloadBits), ...
    'CorrectBlocks',nnz(t.CorrectBlock),'Blocks',height(t), ...
    'Goodput_Mbps',sum(t.Goodput_Mbps)/frame,'EVM_percent',mean(t.EVM_percent), ...
    'MSEToVarianceRatio',mean(t.MSEToVarianceRatio));
   fprintf('RECEIVER_AUDIT_RESULT %s\n',jsonencode(summary));
  end
 end
end
end

function harqRecovery(antennas,numRB,frames,scheme)
% Actual incremental redundancy, with CRC-only acceptance and counted airtime.
% A cohort of four packets is kept on air until all users ACK or four rounds end.
% Already-ACKed layers remain occupied to preserve the common comparison packet.
addpath(pwd,'mu_mimo','v2_mimo');
loaded=load(fullfile('receiver_audit','integrated_receiver',lower(scheme)+"_receiver.mat"),'receiver');
receiver=loaded.receiver;cfg=v3Config(antennas,numRB,"mobile");r=v3Resources(cfg);
for name=string(fieldnames(receiver.TrainingChannelConfig)).'
 assert(isequal(receiver.TrainingChannelConfig.(name),cfg.(name)), ...
  'HARQ receiver configuration mismatch: %s',name);
end
assert(receiver.Scheme==scheme && receiver.TrainingCodeRate==cfg.Rate);
cfg.DecoderDual=false;rvSequence=[0 2 3 1];methods=["LS" "LMMSE" "CNN" "Perfect"];
feedbackGap=r.FrameDuration; % Explicit one-slot ideal-ACK processing interval.
records={};cohorts={};
for frame=1:frames
 baseSeed=100000000+frame;
 [~,payload]=muPayload(cfg,r,muSeed(baseSeed,70));
 buffers=cell(numel(methods),cfg.Users);ack=false(numel(methods),cfg.Users);
 firstACK=zeros(size(ack));errors=repmat(r.PayloadBits,size(ack));
 usedRounds=zeros(numel(methods),1);
 for round=1:numel(rvSequence)
  rv=rvSequence(round);
  current=cfg;current.ChannelSeed=baseSeed;
  current.ChannelInitialTime=(round-1)*(r.FrameDuration+feedbackGap);
  % Independent receiver/RF noise, one time-evolving propagation realization.
  scene=v2Sounding(current,r,baseSeed+10000*(round-1));
  tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,receiver.PriorsTX);
  [weights,power]=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
  assert(power.MaximumPowerError<1e-6);
  grid=r.ReferenceGrid;coded=zeros(r.CodedBits,cfg.Users,'int8');
  for u=1:cfg.Users
   coded(:,u)=r.Codec.encode(payload(:,u),rv);
   layer=grid(:,:,u);
   layer(r.DataIndices)=qammod(coded(r.BitOrder,u),cfg.M,'InputType','bit','UnitAveragePower',true);
   grid(:,:,u)=layer;
  end
  packet=v2Waveform(scene,r,grid,weights,true);
  input=v3Receive(packet,scene,r,40,false,false);oracle=input.Truth;input.Truth=[];
  for k=1:numel(methods)
   if all(ack(k,:)),continue;end % Stop this estimator using CRC only.
   method=methods(k);local=input;
   if method=="Perfect",local.Truth=oracle;end
   [h,v,observed]=v3EstimateRX(local,receiver,r,method);local.Observed=observed;
   % Capture symbols/variance only; the legacy RV0 hard decode is not selected.
   [~,~,~,primary]=v2Decode(local,h,v,payload,cfg,r);
   for u=find(~ack(k,:))
    demapped=v4SymbolLLR(primary.Symbols(:,u),primary.Variance(:,u), ...
        zeros(size(primary.Variance(:,u))),cfg.M);
    llr=zeros(r.CodedBits,1);llr(r.BitOrder)=max(-50,min(50,demapped));
    [decoded,failed,~,buffers{k,u}]=r.Codec.decode(llr,50, ...
        'Layered belief propagation',rv,buffers{k,u});
    errors(k,u)=nnz(decoded~=payload(:,u)); % Scoring only, never ACK control.
    if ~failed,ack(k,u)=true;firstACK(k,u)=round;end
   end
   usedRounds(k)=round;
  end
  for k=1:numel(methods)
   for u=1:cfg.Users
    records{end+1}=struct('Precoder',char(scheme),'BS',antennas,'Frame',frame, ...
     'ChannelSeed',baseSeed,'Estimator',char(methods(k)),'SNR_dB',40,'UE',u, ...
     'RoundBudget',round,'UsedRounds',usedRounds(k),'RV',rv, ...
     'PayloadBits',r.PayloadBits,'Errors',errors(k,u),'CRCFailed',~ack(k,u), ...
     'FirstACKRound',firstACK(k,u),'CorrectBlock',ack(k,u) && errors(k,u)==0); %#ok<AGROW>
   end
   used=usedRounds(k);correct=nnz(ack(k,:) & errors(k,:)==0);
   airtime=used*r.FrameDuration;elapsed=airtime+max(0,used-1)*feedbackGap;
   cohorts{end+1}=struct('Precoder',char(scheme),'BS',antennas,'Frame',frame, ...
    'Estimator',char(methods(k)),'RoundBudget',round,'UsedRounds',used, ...
    'CorrectBlocks',correct,'Blocks',cfg.Users,'PayloadBits',r.PayloadBits*cfg.Users, ...
    'Errors',sum(errors(k,:)),'DeliveredBits',correct*r.PayloadBits, ...
    'Airtime_s',airtime,'Elapsed_s',elapsed,'FeedbackGap_s',feedbackGap, ...
    'InitialCodeRate',cfg.Rate,'PayloadPerTransmittedBit',r.PayloadBits/(used*r.CodedBits)); %#ok<AGROW>
  end
  if ~isfolder('harq_recovery'),mkdir('harq_recovery');end
  prefix=fullfile('harq_recovery',scheme+"_BS"+antennas);
  writetable(struct2table(vertcat(records{:})),prefix+"_blocks.csv");
  C=struct2table(vertcat(cohorts{:}));writetable(C,prefix+"_cohorts.csv");
  for method=methods
   t=C(string(C.Estimator)==method & C.RoundBudget==round,:);
   result=struct('Precoder',scheme,'BS',antennas,'Estimator',method,'SNR_dB',40, ...
    'Frames',frame,'RoundBudget',round,'BER',sum(t.Errors)/sum(t.PayloadBits), ...
    'CorrectBlocks',sum(t.CorrectBlocks),'Blocks',sum(t.Blocks), ...
    'BLER',1-sum(t.CorrectBlocks)/sum(t.Blocks), ...
    'GoodputAirTime_Mbps',sum(t.DeliveredBits)/sum(t.Airtime_s)/1e6, ...
    'GoodputElapsed_Mbps',sum(t.DeliveredBits)/sum(t.Elapsed_s)/1e6, ...
    'MeanRounds',mean(t.UsedRounds));
   fprintf('HARQ_RECOVERY_RESULT %s\n',jsonencode(result));
  end
 end
end
settings=struct('Config',cfg,'RVSequence',rvSequence,'SourceReceiverRun',43193721852, ...
 'FrameCount',frames,'FeedbackGap_s',feedbackGap,'SeedOffset',100000000, ...
 'Feedback','Ideal CRC ACK/NACK, one-slot gap; no feedback-channel errors modeled', ...
 'Scheduling','Four-packet cohort; keep ACKed layers occupied; stop each estimator on all CRC ACKs', ...
 'Channel','One seed per cohort; InitialTime advanced; independent RF/receiver noise per round');
save(prefix+"_settings.mat",'settings');
final=C(string(C.Estimator)=="CNN" & C.RoundBudget==numel(rvSequence),:);
gate=struct('Precoder',scheme,'BS',antennas,'CorrectBlocks',sum(final.CorrectBlocks), ...
 'Blocks',sum(final.Blocks),'AllCNNBlocksDelivered',all(final.CorrectBlocks==final.Blocks), ...
 'GoodputElapsed_Mbps',sum(final.DeliveredBits)/sum(final.Elapsed_s)/1e6, ...
 'Scope','Small acceptance sample; passing does not establish a low outage probability');
fprintf('HARQ_ACCEPTANCE_RESULT %s\n',jsonencode(gate));
assert(gate.AllCNNBlocksDelivered, ...
 'HARQ executed, but the CNN reliability acceptance gate was not met. Inspect saved results.');
end

function firstPassJoint(antennas,numRB,frames,scheme)
% Receiver-only comparison: identical H estimate, data, pilots, power and noise.
addpath(pwd,'mu_mimo','v2_mimo');
loaded=load(fullfile('receiver_audit','integrated_receiver',lower(scheme)+"_receiver.mat"),'receiver');
receiver=loaded.receiver;cfg=v3Config(antennas,numRB,"mobile");r=v3Resources(cfg);
for name=string(fieldnames(receiver.TrainingChannelConfig)).'
 assert(isequal(receiver.TrainingChannelConfig.(name),cfg.(name)), ...
  'First-pass receiver configuration mismatch: %s',name);
end
assert(receiver.Scheme==scheme && receiver.TrainingCodeRate==cfg.Rate);
assert(cfg.M==512 && cfg.Rate==.75 && r.PayloadBits==71280 && numRB==264);
cfg.DecoderDual=false;cfg.DecoderIterations=50;cfg.DecoderAlgorithm='Layered belief propagation';
cfg.DetectorTimeLimit_s=90;methods=["CNN" "Perfect" "LMMSE" "LS"];
rows={};failures={};ceiling=cfg.Users*r.PayloadBits/r.FrameDuration/1e6;
if ~isfolder('first_pass_joint'),mkdir('first_pass_joint');end
prefix=fullfile('first_pass_joint',scheme+"_BS"+antennas);
settings=struct('Config',cfg,'SourceReceiverRun',43193721852,'FramesRequested',frames, ...
 'SeedOffset',102000000,'SNR_dB',40,'FrameDuration',r.FrameDuration, ...
 'PayloadCeiling_Mbps',ceiling,'HARQ',false,'AddedPilotRE',0, ...
 'ComputeLimit_s_perUE',cfg.DetectorTimeLimit_s, ...
 'Warning','Radio-airtime goodput; detector CPU seconds are recorded separately, not a real-time hardware benchmark');
save(prefix+"_settings.mat",'settings');
for frame=1:frames
 scene=v2Sounding(cfg,r,102000000+frame);
 [grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,receiver.PriorsTX);
 [weights,power]=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
 assert(power.MaximumPowerError<1e-6);
 packet=v2Waveform(scene,r,grid,weights,true);
 input=v3Receive(packet,scene,r,40,false,false);oracle=input.Truth;input.Truth=[];
 for method=methods
  local=input;if method=="Perfect",local.Truth=oracle;end
  [h,v,observed]=v3EstimateRX(local,receiver,r,method);local.Observed=observed;
  for detector=["Linear" "Sphere"]
   cfg.RXDetector=char(detector);clock=tic;
   try
    [errors,failed,detail,primary]=v2Decode(local,h,v,payload,cfg,r);
   catch exception
    if ~strcmp(exception.identifier,'v2Decode:SphereBudget'),rethrow(exception);end
    failure=struct('Precoder',char(scheme),'BS',antennas,'Frame',frame, ...
     'Estimator',char(method),'Detector',char(detector),'Reason',exception.message, ...
     'DetectorElapsed_s',toc(clock));failures{end+1}=failure; %#ok<AGROW>
    writetable(struct2table(vertcat(failures{:})),prefix+"_incomplete.csv");
    fprintf('FIRST_PASS_INCOMPLETE %s\n',jsonencode(failure));
    continue
   end
   seconds=toc(clock);
   for u=1:cfg.Users
    coded=r.Codec.encode(payload(:,u));correct=~failed(u) && errors(u)==0;
    rows{end+1}=struct('Precoder',char(scheme),'BS',antennas,'Frame',frame, ...
     'ChannelSeed',scene.Seed,'SNR_dB',40,'Estimator',char(method), ...
     'Detector',char(detector),'UE',u,'PayloadBits',r.PayloadBits, ...
     'Errors',errors(u),'CRCFailed',failed(u),'CorrectBlock',correct, ...
     'PreLDPC_BER',mean((primary.LLR(:,u)<0)~=coded), ...
     'DetectorElapsed_s',seconds,'JointDetectionSeconds',detail.JointDetectionSeconds(u), ...
     'RadioTransmissions',1,'FrameDuration',r.FrameDuration); %#ok<AGROW>
   end
   T=struct2table(vertcat(rows{:}));writetable(T,prefix+"_blocks.csv");
   t=T(string(T.Estimator)==method & string(T.Detector)==detector,:);
   completeFrames=numel(unique(t.Frame));
   summary=struct('Precoder',scheme,'BS',antennas,'Estimator',method,'Detector',detector, ...
    'FramesCompleted',completeFrames,'FramesRequestedSoFar',frame,'SNR_dB',40, ...
    'BER',sum(t.Errors)/sum(t.PayloadBits),'CorrectBlocks',nnz(t.CorrectBlock),'Blocks',height(t), ...
    'BLER',1-nnz(t.CorrectBlock)/height(t),'PreLDPC_BER',mean(t.PreLDPC_BER), ...
    'Goodput_Mbps',nnz(t.CorrectBlock)*r.PayloadBits/(completeFrames*r.FrameDuration*1e6), ...
    'PayloadCeiling_Mbps',ceiling,'HARQ',false,'SampleIncomplete',completeFrames~=frame);
   fprintf('FIRST_PASS_JOINT_RESULT %s\n',jsonencode(summary));
  end
 end
end
target=T(string(T.Estimator)=="CNN" & string(T.Detector)=="Sphere",:);
gate=struct('Precoder',scheme,'BS',antennas,'ExpectedBlocks',frames*cfg.Users, ...
 'CompletedBlocks',height(target),'CorrectBlocks',nnz(target.CorrectBlock), ...
 'Passed',height(target)==frames*cfg.Users && all(target.CorrectBlock), ...
 'Requirement','All CNN blocks delivered on first transmission; no extra pilots, lower code rate or HARQ');
fprintf('FIRST_PASS_ACCEPTANCE_RESULT %s\n',jsonencode(gate));
assert(gate.Passed,'First-pass reliability acceptance was not met. Keep baseline and failure records.');
end
