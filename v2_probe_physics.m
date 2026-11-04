function v2_probe_physics
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
