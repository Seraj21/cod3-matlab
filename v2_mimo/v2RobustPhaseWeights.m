function [weights,diagnostic]=v2RobustPhaseWeights(h,uncertainty,scene,r,baseline)
% Separate phase-only control; the published V2 baseline remains unchanged.
% Diagonal posterior-error approximation, fixed 20-dB design noise, 4-bit phases.
cfg=scene.Config;W=cfg.BeamWidth;G=r.K/W;times=cfg.DownlinkSymbols;
assert(mod(r.K,W)==0 && isequal(size(h),size(uncertainty)));
block=reshape(double(h(:,times,:,:)),W,G,numel(times),cfg.RFChains,cfg.Users);
H=reshape(permute(mean(block,1),[5 4 2 3 1]),cfg.Users,cfg.RFChains,[]);
P=size(H,3);step=2*pi/2^cfg.PhaseBits;amplitude=1/sqrt(cfg.Users*cfg.RFChains);
phases=amplitude*exp(1i*step*(0:2^cfg.PhaseBits-1));
v=mean(reshape(double(uncertainty(:,times,:,:)),W,G,numel(times),cfg.RFChains,cfg.Users),[1 4 5]);
% With equal constant-modulus weights, aggregate diagonal CSI leakage is mean(v).
eta=reshape(v,1,1,[])+10^(-20/10);
if nargin<5,baseline=v2Weights(h,"PZF",scene,r);end
old=mean(reshape(double(baseline(:,:,:,times)),cfg.RFChains,cfg.Users,W,G,numel(times)),3);
old=reshape(old,cfg.RFChains,cfg.Users,P);
% Exported baseline weights are single: restore the exact double phase lattice.
old=amplitude*exp(1i*step*round(angle(old)/step));
gram=pagemtimes(H,'none',H,'ctranspose');
Z=pagemtimes(H,'ctranspose',pagemldivide(gram,repmat(eye(cfg.Users),1,1,P)),'none');
starts={old,amplitude*exp(1i*step*round(angle(Z)/step)), ...
 amplitude*exp(1i*step*round(angle(permute(conj(H),[2 1 3]))/step))};
B=old;bestCost=inf(1,cfg.Users,P);initial=zeros(1,cfg.Users,P);
for start=1:numel(starts)
 for u=1:cfg.Users
  other=setdiff(1:cfg.Users,u);w=starts{start}(:,u,:);response=pagemtimes(H,w);
  objective=@(a)(sum(abs(a(other,:,:)).^2,1)+eta)./max(abs(a(u,:,:)).^2,eps);
  before=objective(response);if start==1,initial(:,u,:)=before;end
  for iteration=1:12
   changed=false;
   for q=1:cfg.RFChains
    candidates=response+H(:,q,:).*(reshape(phases,1,[],1)-w(q,1,:));
    cost=objective(candidates);[value,index]=min(cost,[],2);
    current=objective(response);improve=value<current-1e-14*max(1,current);
    if ~any(improve,'all'),continue;end
    changed=true;index(~improve)=1;chosen=reshape(phases(index),1,1,[]);
    chosen(~improve)=w(q,1,~improve);
    response=response+H(:,q,:).*(chosen-w(q,1,:));w(q,1,:)=chosen;
   end
   if ~changed,break;end
  end
  cost=objective(response);assert(all(cost<=before+1e-9,'all'));
  better=reshape(cost<bestCost(:,u,:),1,[]);chosen=B(:,u,:);chosen(:,:,better)=w(:,:,better);
  B(:,u,:)=chosen;bestCost(:,u,:)=min(bestCost(:,u,:),cost);
 end
end
assert(all(bestCost<=initial+1e-9,'all'));
B=reshape(B,cfg.RFChains,cfg.Users,G,numel(times));
if cfg.AlignPhase
 for u=1:cfg.Users
  first=B(:,u,:,1);increments=angle(sum(conj(first(:,:,1:end-1)).*first(:,:,2:end),1));
  increments=step*round(increments/step);accumulated=[0 reshape(cumsum(increments,3),1,[])];
  B(:,u,:,1)=first.*reshape(exp(-1i*accumulated),1,1,G);
  for t=2:numel(times)
   phase=angle(sum(conj(B(:,u,:,t-1)).*B(:,u,:,t),1));phase=step*round(phase/step);
   B(:,u,:,t)=B(:,u,:,t).*exp(-1i*phase);
  end
 end
end
power=reshape(sum(abs(B).^2,[1 2]),1,[]);
assert(max(abs(power-1))<1e-8 && max(abs(abs(B)-amplitude),[],'all')<1e-12);
weights=complex(zeros(cfg.RFChains,cfg.Users,r.K,r.L,'single'));
for t=1:numel(times),weights(:,:,:,times(t))=single(repelem(B(:,:,:,t),1,1,W));end
diagnostic=struct('MaximumPowerError',max(abs(power-1)), ...
 'MeanInitialObjective',mean(initial,'all'),'MeanFinalObjective',mean(bestCost,'all'), ...
 'PosteriorPenalty',eta,'SelectedObjective',bestCost,'DesignSNR_dB',20);
end
