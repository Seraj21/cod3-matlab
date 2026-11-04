function packet = muDownlink(scene,r,grid,weights)
cfg = scene.Config;
x = permute(weights,[3 4 1 2]);
x = sum(x.*reshape(single(grid),r.K,r.L,1,cfg.Users),4);
waveform = nrOFDMModulate(r.Carrier,x,'Windowing',0);
[distorted,txPhase,paGain] = muRadio(waveform,cfg,muSeed(scene.Seed,15),true,0);
channel = scene.Channel;
reset(channel);
rx = channel([distorted.*scene.Front.ChannelGain.'; ...
    zeros(scene.Tail,cfg.RFChains,'single')],r.Carrier);
rxPhase = cell(cfg.Users,1);
for u = 1:cfg.Users
    columns = (u-1)*cfg.UERx+(1:cfg.UERx);
    [rx(:,columns),rxPhase{u}] = muRadio(rx(:,columns),cfg,muSeed(scene.Seed,16,u),false,0);
end
H = permute(scene.HDownlink,[3 4 1 2]);
linear = double(weights).*reshape(paGain,cfg.RFChains,1,1,1);
effective = pagemtimes(H,linear);
effective = permute(effective,[3 4 1 2]);
packet = struct('Waveform',double(rx),'Propagation',effective,'TxPhase',txPhase, ...
    'RxPhase',{rxPhase},'TrueOffset',scene.TrueOffset, ...
    'PAPowerRatio',mean(abs(distorted).^2,'all')/max(mean(abs(waveform).^2,'all'),eps));
end
