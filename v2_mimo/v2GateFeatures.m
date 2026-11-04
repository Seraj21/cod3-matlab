function X=v2GateFeatures(samples,noise,layouts)
% Products remove arbitrary oscillator phase while retaining Doppler statistics.
[K,~,R,U]=size(samples);W=layouts{1}.W;G=K/W;T=4;F=6;
z=complex(zeros(F,T,R,U,G));
noise=noise(:).';if isscalar(noise),noise=repmat(noise,1,R);end
relative=zeros(G,R,U);
for u=1:U
 p=layouts{u};patch=permute(reshape(samples(:,:,:,u),W,G,p.L,R),[1 3 4 2]);
 obs=patch(p.Freq,p.Times,:,:);
 scale=sqrt(max(mean(abs(obs).^2,[1 2])-reshape(noise,1,1,R,1), ...
     max(.1*reshape(noise,1,1,R,1),1e-5)));
 z(:,:,:,u,:)=reshape(obs./scale,F,T,R,1,G);
 relative(:,:,u)=reshape(reshape(noise,1,1,R,1)./scale.^2,R,G).';
end
C=R*U;parts=cell(8,1);parts{1}=reshape(abs(z).^2,F,T,C,G);
at=2;
for lag=1:3
 product=complex(zeros(size(z)));
 product(:,lag+1:T,:,:,:)=z(:,lag+1:T,:,:,:).*conj(z(:,1:T-lag,:,:,:));
 parts{at}=reshape(real(product),F,T,C,G);
 parts{at+1}=reshape(imag(product),F,T,C,G);at=at+2;
end
relative=permute(log10(max(relative,1e-6)),[4 5 2 3 1]);
parts{8}=reshape(repmat(relative,F,T,1,1,1),F,T,C,G);
X=single(cat(3,parts{:}));
end
