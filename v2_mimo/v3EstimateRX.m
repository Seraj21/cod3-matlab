function [h,v,observed]=v3EstimateRX(input,receiver,r,method)
% Retain the original learned feature operator for LMMSE/CNN.
% Only the standalone LS comparator may use its independently selected operator.
if method=="LS" && isfield(receiver,'LSInterpolation')
 r.DownlinkLayout=v3LSLayouts(r,receiver.LSInterpolation);
 receiver.PriorsRX=receiver.PriorsLS;
end
[h,v,observed]=v2EstimateRX(input,receiver,r,method);
if isfield(r,'PilotTracking') && method~="Perfect"
 [h,v]=pilotPosterior(observed,h,input,r);
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
