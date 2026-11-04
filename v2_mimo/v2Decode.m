function [errors,failed,details]=v2Decode(input,h,variance,payload,cfg,r)
% Both candidates use receiver information only. CRC selects a successful retry.
users=true(1,cfg.Users);
if isfield(cfg,'ActiveUsers'),users(:)=false;users(cfg.ActiveUsers)=true;end
[bits,failed,details]=pass(input,h,variance,cfg,r,false,users);
failed(~users)=false;
firstFailed=failed;
if cfg.DecoderDual && any(failed)
 [retry,retryFailed,retryDetails]=pass(input,h,variance,cfg,r,true,failed);
 accepted=failed & ~retryFailed;
 bits(:,accepted)=retry(:,accepted);failed(accepted)=false;
 details.MeanEffectiveVariance(accepted)=retryDetails.MeanEffectiveVariance(accepted);
 details.SRSBeamAccepted=accepted;
else
 details.SRSBeamAccepted=false(1,cfg.Users);
end
details.Retried=firstFailed;
errors=sum(bits~=payload,1);
end
function [decoded,failed,details]=pass(input,h,channelVariance,cfg,r,fixedBeam,users)
decoded=zeros(r.PayloadBits,cfg.Users,'int8');failed=true(1,cfg.Users);
evm=zeros(1,cfg.Users);
for u=find(users)
 rows=(u-1)*cfg.UERx+(1:cfg.UERx);
 if fixedBeam,rows=rows(1);end % The first unitary receive beam is the SRS beam.
 R=numel(rows);
 G=permute(h(:,:,rows,:),[3 4 1 2]);
 Y=reshape(permute(input.Observed(:,:,rows),[3 1 2]),R,1,r.K,r.L);
 disturbance=v2NoiseCovariance(input.Observed(:,:,rows),h(:,:,rows,:), ...
     channelVariance(:,:,rows,:),input.Noise(rows),r);
 covariance=pagemtimes(G,'none',G,'ctranspose')+disturbance;
 desired=G(:,u,:,:);w=pagemldivide(covariance,desired);
 alpha=pagemtimes(w,'ctranspose',desired,'none');
 alphaPower=max(abs(alpha).^2,eps);
 equalized=reshape(pagemtimes(w,'ctranspose',Y,'none')./max(real(alpha),eps),r.K,r.L);
 response=pagemtimes(w,'ctranspose',G,'none');
 coupling=permute(reshape(response./max(real(alpha),eps),cfg.Users,r.K,r.L),[2 3 1]);
 response(:,u,:,:)=0;
 interference=reshape(sum(abs(response).^2,2)./alphaPower,r.K,r.L);
 postNoise=pagemtimes(w,'ctranspose',pagemtimes(disturbance,w),'none');
 postNoise=reshape(max(real(postNoise),0)./alphaPower,r.K,r.L);
 softVariance=interference+postNoise;
 reference=r.PTRS(:,:,u);indices=r.PTRSIndices{u};
 [~,times]=ind2sub([r.K r.L],indices);
 phasor=accumarray(times,equalized(indices).*conj(reference(indices)),[r.L 1]);
 valid=find(abs(phasor)>0);
 if numel(valid)>1
  phase=interp1(valid,unwrap(angle(phasor(valid))),min(valid(end),max(valid(1),(1:r.L).')),'linear');
  equalized=equalized.*exp(-1i*phase.');
 end
 % Remove all known pilot contributions before estimating unmodelled ICI/ISI.
 expected=sum(coupling.*r.DMRS,3);
 pilotVariance=postNoise;
 mask=abs(r.DMRS(:,:,u))>0;times=find(any(mask,1));extra=zeros(numel(times),1);
 for k=1:numel(times)
  l=times(k);f=find(mask(:,l));
  extra(k)=max(0,mean(abs(equalized(f,l)-expected(f,l)).^2-pilotVariance(f,l)));
 end
 extra=interp1(times,extra,min(times(end),max(times(1),(1:r.L).')),'linear');
 effectiveVariance=max(1e-7,softVariance(r.DataIndices)+extra(r.DataTimes));
 llr=qamdemod(equalized(r.DataIndices),cfg.M,'OutputType','approxllr', ...
     'UnitAveragePower',true,'NoiseVariance',1);
 llr=reshape(reshape(llr,cfg.Qm,[])./reshape(effectiveVariance,1,[]),[],1);
 restored=zeros(r.CodedBits,1);restored(r.BitOrder)=llr;
 [decoded(:,u),failed(u)]=r.Codec.decode(max(-50,min(50,restored)));
 evm(u)=mean(effectiveVariance);
end
details=struct('MeanEffectiveVariance',evm);
end
