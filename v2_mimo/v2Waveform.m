function packet=v2Waveform(scene,r,grid,weights,truthNeeded)
if nargin<5,truthNeeded=true;end
cfg=scene.Config;
x=sum(permute(weights,[3 4 1 2]).*reshape(single(grid),r.K,r.L,1,cfg.Users),4);
wave=nrOFDMModulate(r.Carrier,x,'Windowing',0);
[distorted,txPhase,paGain]=muRadio(wave,cfg,muSeed(scene.Seed,15),true,0);
channel=v2Channel(cfg,scene.Front,cfg.SampleRate,scene.Seed);
channel.ChannelResponseOutput='path-gains';
input=[distorted.*scene.Front.ChannelGain.';zeros(scene.Tail,cfg.RFChains,'single')];
if truthNeeded
 [rx,pg,st]=channel(input);pf=getPathFilters(channel);
 offset=nrPerfectTimingEstimate(pg,pf);
 H=nrPerfectChannelEstimate(r.Carrier,pg,pf,offset,st,'CyclicPrefixFraction',cfg.CPFraction);
 H=double(H(:,1:r.L,:,:)).*reshape(scene.Front.ChannelGain,1,1,1,cfg.RFChains);
 H=permute(pagemtimes(permute(H,[3 4 1 2]), ...
     double(weights).*reshape(paGain,cfg.RFChains,1,1,1)),[3 4 1 2]);
else
 rx=channel(input);H=[];offset=scene.TrueOffset;
end
rxPhase=cell(cfg.Users,1);
for u=1:cfg.Users
 columns=(u-1)*cfg.UERx+(1:cfg.UERx);
 [rx(:,columns),rxPhase{u}]=muRadio(rx(:,columns),cfg,muSeed(scene.Seed,16,u),false,0);
end
packet=struct('Waveform',double(rx),'Propagation',H,'TxPhase',txPhase, ...
 'RxPhase',{rxPhase},'TrueOffset',offset,'PAGain',paGain);
end
