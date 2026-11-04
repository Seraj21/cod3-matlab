function state=v2CEBase(samples,noise,layouts,models)
[K,L,R,U]=size(samples);W=layouts{1}.W;G=K/W;
noise=noise(:).';if isscalar(noise),noise=repmat(noise,1,R);end
state=struct('LS',complex(zeros(K,L,R,U)),'Wiener',complex(zeros(K,L,R,U)), ...
 'Pilot',samples,'Scale',zeros(G,R,U),'RelativeNoise',zeros(G,R,U), ...
 'VarianceLS',zeros(K,L,R,U),'VarianceWiener',zeros(K,L,R,U));
state.RawVarianceWiener=state.VarianceWiener;
state.RawVarianceLS=state.VarianceLS;
for u=1:U
 p=layouts{u};separate=~isfield(models,p.Label);
 if separate,groups=num2cell(1:R);else,groups={1:R};end
 for group=1:numel(groups)
  branches=groups{group};key=p.Label;if separate,key=key+"_R"+branches;end
  m=models.(key);Rb=numel(branches);
  patches=reshape(permute(reshape(double(samples(:,:,branches,u)),W,G,L,Rb), ...
      [1 3 2 4]),W*L,[]).';
  obs=patches(:,p.Observed);n=repelem(noise(branches),G).'*p.NoiseGain;
  scale=sqrt(max(mean(abs(obs).^2,2)-n,max(.1*n,1e-5)));
  relative=n./scale.^2;inverse=1./(m.D+relative);
  ls=obs*p.Linear;
  wiener=(((((obs./scale-m.MeanX)*m.V).*inverse)*m.Cross)+m.MeanY).*scale;
  vls=scale.^2.*m.LSBias+n.*m.LSNoise;
  vw=max(1e-10,scale.^2.*(m.YY-inverse*abs(m.Cross).^2));
  state.LS(:,:,branches,u)=restore(ls,W,L,G,Rb);
  state.Wiener(:,:,branches,u)=restore(wiener,W,L,G,Rb);
  state.RawVarianceLS(:,:,branches,u)=restore(vls,W,L,G,Rb);
  state.RawVarianceWiener(:,:,branches,u)=restore(vw,W,L,G,Rb);
  state.VarianceLS(:,:,branches,u)=restore(vls,W,L,G,Rb)*m.VarianceFactor.LS;
  state.VarianceWiener(:,:,branches,u)=restore(vw,W,L,G,Rb)*m.VarianceFactor.LMMSE;
  state.Scale(:,branches,u)=reshape(scale,G,Rb);
  state.RelativeNoise(:,branches,u)=reshape(relative,G,Rb);
 end
end
end
function h=restore(rows,W,L,G,R)
h=reshape(permute(reshape(rows.',W,L,G,R),[1 3 2 4]),W*G,L,R);
end
