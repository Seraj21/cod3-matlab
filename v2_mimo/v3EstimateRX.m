function [h,v,observed]=v3EstimateRX(input,receiver,r,method)
% Retain the original learned feature operator for LMMSE/CNN.
% Only the standalone LS comparator may use its independently selected operator.
if method=="LS" && isfield(receiver,'LSInterpolation')
 r.DownlinkLayout=v3LSLayouts(r,receiver.LSInterpolation);
 receiver.PriorsRX=receiver.PriorsLS;
end
[h,v,observed]=v2EstimateRX(input,receiver,r,method);
if isfield(r,'PilotTracking') && method~="Perfect"
 if r.OrthogonalDMRS
  [h,v]=separatedPosterior(observed,h,input,r);
 else
  [h,v]=pilotPosterior(observed,h,input,r);
 end
end
end

function [h,v]=pilotPosterior(y,h,input,r)
% Condition a local channel prior on the raw, simultaneous pilot equations.
% A beam is constant across this group and downlink interval by construction.
% Use the frozen estimator for the local mean, and a TDL-C covariance for
% the residual. No transmitted data, decoded bits or realized path gains enter.
p=r.PilotTracking;W=p.Width;L=r.L;times=find(any(r.ReferenceGrid~=0,[1 3]));
[f,t]=ndgrid(0:W-1,1:L);f=f(:);t=t(:);
time=p.Time(t);time=time(:);
Ct=besselj(0,2*pi*p.Doppler*(time-time.'));
power=10.^(p.Gains(:)/10);power=power/sum(power);
v=zeros(size(h));
for ue=1:numel(input.TimingOffset)
 rows=(ue-1)*4+(1:4);
 delays=p.Delays(:)+(p.FilterDelay-input.TimingOffset(ue))/r.OFDM.SampleRate;
 F=exp(-2i*pi*(0:W-1).'*p.Spacing*delays.');
 Cf=(F.*power.')*F';C=Cf(f+1,f+1).*Ct;
 for first=1:W:r.K
  bins=first:first+W-1;
  for pair=[1 3]
   ports=pair:pair+1;
   pilots=reshape(r.ReferenceGrid(bins,:,ports),W*L,2);
   at=find(any(pilots~=0,2));
   symbols=pilots(at,:);
   z=reshape(y(bins,:,rows),W*L,4);z=z(at,:);
   prior=reshape(mean(h(bins,times,rows,ports),[1 2]),4,2).';
   residual=z-symbols*prior;
   channelPower=max(mean(abs(prior).^2,2),.02*mean(abs(z).^2,'all'));
   channelPower=max(channelPower,mean(abs(z).^2,'all')/8+eps);
   loading=max(mean(input.Noise(rows)),3e-4*mean(abs(z).^2,'all'))+eps;
   S=C(at,at).*((symbols.*channelPower.')*symbols')+loading*eye(numel(at));
   S=(S+S')/2;R=chol(S,'lower');
   whitened=R\residual;
   for k=1:2
    cross=(C(:,at).*conj(symbols(:,k)).')*channelPower(k);
    gain=cross/R';
    value=prior(k,:)+gain*whitened;
    h(bins,:,rows,ports(k))=reshape(value,W,L,4);
    post=max(loading/16,channelPower(k)-sum(abs(gain).^2,2));
    v(bins,:,rows,ports(k))=repmat(reshape(post,W,L),1,1,4);
   end
  end
 end
end
end

function [h,v]=separatedPosterior(y,h,input,r)
% Every pilot is isolated at one RE. After division by that pilot all
% frequency groups share the same sampling matrix, allowing batched solves.
p=r.PilotTracking;W=p.Width;G=r.K/W;L=r.L;
[f,t]=ndgrid(0:W-1,1:L);f=f(:);time=p.Time(t(:));time=time(:);
Ct=besselj(0,2*pi*p.Doppler*(time-time.'));
power=10.^(p.Gains(:)/10);power=power/sum(power);
v=zeros(size(h));
for ue=1:numel(input.TimingOffset)
 rows=(ue-1)*4+(1:4);
 delays=p.Delays(:)+(p.FilterDelay-input.TimingOffset(ue))/r.OFDM.SampleRate;
 F=exp(-2i*pi*(0:W-1).'*p.Spacing*delays.');
 Cf=(F.*power.')*F';C=Cf(f+1,f+1).*Ct;
 for port=1:4
  ref=reshape(permute(reshape(r.ReferenceGrid(:,:,port),W,G,L),[1 3 2]),W*L,G);
  period=r.DMRSLegacyLayouts{port}.W/W;assert(period>=1 && period==floor(period));
  samples=reshape(permute(reshape(y(:,:,rows),W,G,L,4),[1 3 2 4]),W*L,G,4);
  localMean=reshape(mean(reshape(h(:,6:12,rows,port),W,G,7,4),[1 3]),G,4);
  fitted=complex(zeros(W*L,G,4));variance=zeros(W*L,G,4);
  for phase=1:period
   groups=phase:period:G;at=find(ref(:,phase));
   assert(isequal(ref(:,groups)~=0,repmat(ref(:,phase)~=0,1,numel(groups))));
   z=reshape(samples(at,groups,:)./ref(at,groups),numel(at),[]);
   prior=reshape(localMean(groups,:),1,[]);scale=max(mean(abs(z).^2,1),eps);
   noiseRatio=max(mean(input.Noise(rows))/(4*median(scale)),3e-4);
   S=C(at,at)+noiseRatio*eye(numel(at));R=chol((S+S')/2,'lower');
   gain=C(:,at)/R';
   fitted(:,groups,:)=reshape(prior+gain*(R\(z-prior)),W*L,numel(groups),4);
   variance(:,groups,:)=reshape(max(noiseRatio/16,1-sum(abs(gain).^2,2)).*scale,W*L,numel(groups),4);
  end
  h(:,:,rows,port)=reshape(permute(reshape(fitted,W,L,G,4),[1 3 2 4]),r.K,L,4);
  v(:,:,rows,port)=reshape(permute(reshape(variance,W,L,G,4),[1 3 2 4]),r.K,L,4);
 end
end
end
