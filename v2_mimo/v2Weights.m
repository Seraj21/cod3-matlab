function [weights,diagnostic]=v2Weights(h,scheme,scene,r)
% Batched frequency-selective MR/ZF/4-bit phase-only ZF, equal total power.
cfg=scene.Config;W=cfg.BeamWidth;G=r.K/W;times=cfg.DownlinkSymbols;
assert(mod(r.K,W)==0);
block=reshape(double(h(:,times,:,:)),W,G,numel(times),cfg.RFChains,cfg.Users);
H=reshape(permute(mean(block,1),[5 4 2 3 1]),cfg.Users,cfg.RFChains,[]);
if scheme=="MR"
 B=permute(conj(H),[2 1 3]);
else
 gram=pagemtimes(H,'none',H,'ctranspose');
 invGram=pagemldivide(gram,repmat(eye(cfg.Users),1,1,size(gram,3)));
 B=pagemtimes(H,'ctranspose',invGram,'none');
end
B=B./max(sqrt(sum(abs(B).^2,1)),eps)/sqrt(cfg.Users);
initial=zeros(1,size(H,3));final=initial;
if scheme=="PZF"
 step=2*pi/2^cfg.PhaseBits;
 amplitude=1/sqrt(cfg.Users*cfg.RFChains);
 B=amplitude*exp(1i*step*round(angle(B)/step));
 phases=amplitude*exp(1i*step*(0:2^cfg.PhaseBits-1));
 eta=1e-6*max(mean(abs(H).^2,[1 2]),eps);
 for u=1:cfg.Users
  other=setdiff(1:cfg.Users,u);
  w=B(:,u,:);response=pagemtimes(H,w);
  objective=@(a)(sum(abs(a(other,:,:)).^2,1)+eta)./max(abs(a(u,:,:)).^2,eps);
  initial=initial+reshape(objective(response),1,[]);
  for iteration=1:2
   for q=1:cfg.RFChains
    candidates=response+H(:,q,:).*(reshape(phases,1,[],1)-w(q,1,:));
    cost=objective(candidates);
    [best,index]=min(cost,[],2);
    improve=best<objective(response)-1e-14*max(1,objective(response));
    index(~improve)=1;
    chosen=reshape(phases(index),1,1,[]);
    chosen(~improve)=w(q,1,~improve);
    response=response+H(:,q,:).*(chosen-w(q,1,:));
    w(q,1,:)=chosen;
   end
  end
  B(:,u,:)=w;
  final=final+reshape(objective(response),1,[]);
 end
end
B=reshape(B,cfg.RFChains,cfg.Users,G,numel(times));
if cfg.AlignPhase
 for u=1:cfg.Users
  first=B(:,u,:,1);
  increments=angle(sum(conj(first(:,:,1:end-1)).*first(:,:,2:end),1));
  if scheme=="PZF",increments=step*round(increments/step);end
  accumulated=[0 reshape(cumsum(increments,3),1,[])];
  B(:,u,:,1)=first.*reshape(exp(-1i*accumulated),1,1,G);
  for t=2:numel(times)
   phase=angle(sum(conj(B(:,u,:,t-1)).*B(:,u,:,t),1));
   if scheme=="PZF",phase=step*round(phase/step);end
   B(:,u,:,t)=B(:,u,:,t).*exp(-1i*phase);
  end
 end
end
power=reshape(sum(abs(B).^2,[1 2]),1,[]);
weights=complex(zeros(cfg.RFChains,cfg.Users,r.K,r.L,'single'));
for t=1:numel(times)
 weights(:,:,:,times(t))=single(repelem(B(:,:,:,t),1,1,W));
end
assert(max(abs(power-1))<1e-8);
if scheme=="PZF",assert(all(final<=initial+1e-9));end
diagnostic=struct('MaximumPowerError',max(abs(power-1)), ...
 'MeanInitialObjective',mean(initial),'MeanFinalObjective',mean(final));
end
