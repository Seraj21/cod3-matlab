function [errors,failed,details,primary]=v2Decode(input,h,variance,payload,cfg,r)
% Both candidates use receiver information only. CRC selects a successful retry.
users=true(1,cfg.Users);
if isfield(cfg,'ActiveUsers'),users(:)=false;users(cfg.ActiveUsers)=true;end
capture=nargout>3;
[bits,failed,details,primary]=pass(input,h,variance,cfg,r,false,users,capture);
failed(~users)=false;
firstFailed=failed;
joint=isfield(cfg,'RXDetector') && string(cfg.RXDetector)=="Sphere";
if cfg.DecoderDual && any(failed) && ~joint
 [retry,retryFailed,retryDetails]=pass(input,h,variance,cfg,r,true,failed,false);
 accepted=failed & ~retryFailed;
 bits(:,accepted)=retry(:,accepted);failed(accepted)=false;
 details.MeanEffectiveVariance(accepted)=retryDetails.MeanEffectiveVariance(accepted);
 details.SRSBeamAccepted=accepted;
else
 details.SRSBeamAccepted=false(1,cfg.Users);
end
details.Retried=firstFailed;
errors=sum(bits~=payload,1);
if capture,primary.Decoded=bits;end
end
function [decoded,failed,details,primary]=pass(input,h,channelVariance,cfg,r,fixedBeam,users,capture)
decoded=zeros(r.PayloadBits,cfg.Users,'int8');failed=true(1,cfg.Users);
evm=zeros(1,cfg.Users);
detectionSeconds=zeros(1,cfg.Users);
joint=isfield(cfg,'RXDetector') && string(cfg.RXDetector)=="Sphere";
primary=struct;
if capture
 primary.Symbols=zeros(numel(r.DataIndices),cfg.Users,'like',input.Observed);
 primary.Variance=zeros(numel(r.DataIndices),cfg.Users);
 primary.ChannelError=zeros(numel(r.DataIndices),cfg.Users);
 primary.LLR=zeros(r.CodedBits,cfg.Users);
 primary.PTRSAmplitude=ones(r.L,cfg.Users);
end
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
 phase=zeros(r.L,1);
 if capture
  % Diagnostic only: expose the gain measured from known isolated PTRS.
  % Do not alter production symbols, channel estimates or LLR variance here.
  energy=accumarray(times,abs(reference(indices)).^2,[r.L 1]);
  amplitude=abs(phasor)./max(energy,eps);
  if numel(valid)>1
   primary.PTRSAmplitude(:,u)=interp1(valid,amplitude(valid), ...
       min(valid(end),max(valid(1),(1:r.L).')),'linear');
  elseif numel(valid)==1
   primary.PTRSAmplitude(:,u)=amplitude(valid);
  end
 end
 if numel(valid)>1
  phase=interp1(valid,unwrap(angle(phasor(valid))),min(valid(end),max(valid(1),(1:r.L).')),'linear');
  equalized=equalized.*exp(-1i*phase.');
  % PTRS removes residual channel phase omitted by h. The normalized desired
  % response stays one; rotating coupling again would double-count that phase.
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
 if joint
  [llr,detectionSeconds(u)]=jointLLR(input.Observed(:,:,rows),G,disturbance,phase,u,cfg,r);
 else
  llr=qamdemod(equalized(r.DataIndices),cfg.M,'OutputType','approxllr', ...
      'UnitAveragePower',true,'NoiseVariance',1);
  llr=reshape(reshape(llr,cfg.Qm,[])./reshape(effectiveVariance,1,[]),[],1);
 end
 restored=zeros(r.CodedBits,1);restored(r.BitOrder)=llr;
 restored=max(-50,min(50,restored));
 iterations=20;algorithm='Normalized min-sum';
 if isfield(cfg,'DecoderIterations'),iterations=cfg.DecoderIterations;end
 if isfield(cfg,'DecoderAlgorithm'),algorithm=cfg.DecoderAlgorithm;end
 [decoded(:,u),failed(u)]=r.Codec.decode(restored,iterations,algorithm);
 if capture
  % Project the desired-channel error through the same receive combiner.
  squaredWeights=abs(w).^2;
  errorVariance=permute(channelVariance(:,:,rows,u),[3 4 1 2]);
  thermal=reshape(input.Noise(rows),R,1);
  projected=reshape(sum(squaredWeights.*errorVariance,1)./alphaPower,r.K,r.L);
  thermal=reshape(sum(squaredWeights.*thermal,1)./alphaPower,r.K,r.L);
  headroom=max(0,postNoise-thermal);
  projected=min(max(projected,0),headroom);
  primary.Symbols(:,u)=equalized(r.DataIndices);
  primary.Variance(:,u)=effectiveVariance;
  primary.ChannelError(:,u)=projected(r.DataIndices);
  primary.LLR(:,u)=restored;
 end
 evm(u)=mean(effectiveVariance);
end
details=struct('MeanEffectiveVariance',evm,'JointDetectionSeconds',detectionSeconds);
end

function [llr,seconds]=jointLLR(observed,G,disturbance,phase,user,cfg,r)
% Joint finite-alphabet detection uses this UE's antennas only, never payload.
% Whiten BOTH y and H so SphereDecoder's unscaled distance is a noise metric.
R=size(observed,3);assert(R>=cfg.Users);
constellation=qammod((0:cfg.M-1).',cfg.M,'UnitAveragePower',true);
bitTable=zeros(cfg.M,cfg.Qm);
for bit=1:cfg.Qm,bitTable(:,bit)=bitget(uint16((0:cfg.M-1).'),cfg.Qm-bit+1);end
detector=comm.SphereDecoder('Constellation',constellation,'BitTable',bitTable, ...
    'DecisionType','Soft','InitialRadius','ZF solution');
clock=tic;limit=Inf;
if isfield(cfg,'DetectorTimeLimit_s'),limit=cfg.DetectorTimeLimit_s;end
G=reshape(G,R,cfg.Users,r.K*r.L);C=reshape(disturbance,R,R,r.K*r.L);
y=reshape(observed,r.K*r.L,R);N=numel(r.DataIndices);llr=zeros(N*cfg.Qm,1);
chunk=16;
for first=1:chunk:N
 selected=first:min(first+chunk-1,N);count=numel(selected);
 yw=complex(zeros(count,R));hw=complex(zeros(count,cfg.Users,R));
 for n=1:count
  idx=r.DataIndices(selected(n));l=r.DataTimes(selected(n));
  covariance=C(:,:,idx);covariance=(covariance+covariance')/2;
  [factor,notPD]=chol(covariance,'lower');
  if notPD
   floorValue=max(1e-12,1e-9*real(trace(covariance))/R);
   factor=chol(covariance+floorValue*eye(R),'lower');
  end
  channel=factor\G(:,:,idx);
  symbol=factor\(y(idx,:).'*exp(-1i*phase(l)));
  yw(n,:)=symbol.';hw(n,:,:)=reshape(channel.',1,cfg.Users,R);
 end
 if count~=chunk,release(detector);end
 raw=detector(yw,hw);
 % SphereDecoder uses log P(1)/P(0); LDPC/qamdemod use log P(0)/P(1).
 values=-raw(:,user);
 assert(all(isfinite(values)),'Joint detector produced non-finite LLRs');
 llr((first-1)*cfg.Qm+(1:count*cfg.Qm))=values;
 if toc(clock)>limit
  error('v2Decode:SphereBudget','Joint detector exceeded %.0f s for one UE; no partial BER is reported.',limit);
 end
end
seconds=toc(clock);
end
