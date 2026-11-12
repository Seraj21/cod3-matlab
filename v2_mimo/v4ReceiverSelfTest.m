function v4ReceiverSelfTest
% Check all Gray-labelled bit positions against the Communications Toolbox.
M=512;Qm=log2(M);
symbols=qammod((0:M-1).',M,'UnitAveragePower',true)+(.037+.016i);
for noise=[.014 .2 1]
 variance=repmat(noise,M,1);
 actual=v4SymbolLLR(symbols,variance,zeros(M,1),M);
 expected=qamdemod(symbols,M,'UnitAveragePower',true,'OutputType','llr','NoiseVariance',noise);
 assert(max(abs(actual-expected))<1e-8);
end
edge=v4SymbolLLR(symbols,repmat(.001,M,1),repmat(1e8,M,1),M);
assert(all(isfinite(edge)));
% Independent check of SphereDecoder's opposite LLR sign and whitening scale.
constellation=qammod((0:511).',512,'UnitAveragePower',true);
bitTable=zeros(512,9);
for bit=1:9,bitTable(:,bit)=bitget(uint16((0:511).'),10-bit);end
detector=comm.SphereDecoder('Constellation',constellation,'BitTable',bitTable, ...
 'DecisionType','Soft','InitialRadius','ZF solution');
gain=[1.1-.25i .8+.1i];noise=.03;probe=constellation(1:17:end)+(.013-.021i);
probe=[probe flipud(probe)];y=probe.*gain;
actual=-detector(y/sqrt(noise),reshape(diag(gain)/sqrt(noise),1,2,2));
for stream=1:2
 expected=qamdemod(probe(:,stream),512,'UnitAveragePower',true,'OutputType','approxllr', ...
  'NoiseVariance',noise/abs(gain(stream))^2);
 assert(max(abs(actual(:,stream)-expected))<1e-8,'Sphere LLR sign, mapping or noise scale is incorrect');
end
release(detector);
% Exercise the full joint receiver path with four coupled streams and CPE.
cfg=v3Config(32,12);cfg.M=16;cfg.Qm=4;cfg.DecoderDual=false;
cfg.RXDetector='Sphere';cfg.DecoderIterations=50;cfg.DecoderAlgorithm='Layered belief propagation';
cfg.DetectorTimeLimit_s=60;r=v3Resources(cfg);
[grid,payload]=muPayload(cfg,r,71000499);
A=eye(4)+.12i*ones(4)+.08*diag(ones(3,1),1);
h=repmat(reshape(repmat(A,4,1),1,1,16,4),r.K,r.L,1,1);
input=struct('Observed',sum(h.*reshape(grid,r.K,r.L,1,4),4)*exp(.29i), ...
 'Noise',1e-3*ones(1,16));
[errors,failed]=v2Decode(input,h,zeros(size(h)),payload,cfg,r);
assert(all(errors==0 & ~failed),'Joint receiver failed coupled-stream noiseless round trip');
% Compare every RV against the toolbox's independent standard-QAM matcher.
for setup=[512 1024 .5;12000 16000 .75].'
 for qm=[4 8]
  codec=ExperimentalLDPC(setup(1),setup(2),qm,setup(3));
  stream=RandStream('mt19937ar','Seed',71000400+qm+setup(1));
  data=int8(randi(stream,[0 1],codec.PayloadBits,1));
  attached=nrCRCEncode(data,codec.Info.CRC);
  mother=nrLDPCEncode(nrCodeBlockSegmentLDPC(attached,codec.Info.BGN),codec.Info.BGN);
  modulation=sprintf('%dQAM',2^qm);combined=[];
  for rv=[0 2 3 1]
   coded=codec.encode(data,rv);
   expected=nrRateMatchLDPC(mother,codec.CodedBits,rv,modulation,1);
   assert(isequal(coded,expected),'HARQ rate matching differs from NR toolbox');
   soft=5*(1-2*double(coded));
   recovered=codec.recover(soft,rv);
   expected=nrRateRecoverLDPC(soft,codec.PayloadBits,setup(3),rv,modulation,1);
   assert(max(abs(recovered(~codec.FillerMask)-expected(~codec.FillerMask)))<1e-12);
   previous=combined;
   [decoded,failed,~,combined]=codec.decode(soft,50,'Layered belief propagation',rv,previous);
   if isempty(previous),previous=zeros(size(combined));end
   assert(isequal(combined,previous+recovered),'HARQ lost or duplicated prior LLRs');
  end
  assert(~failed && isequal(data,decoded));
  assert(isequal(codec.encode(data),codec.encode(data,0)),'Legacy RV0 changed');
 end
end
% Verify layered BP can decode the identical Qm=9 rate-matched stream.
for numRB=[66 264]
 cfg=v3Config(32,numRB);r=v3Resources(cfg);
 assert(r.PayloadBits==8*floor(.75*r.CodedBits/8) && cfg.Rate==.75);
 stream=RandStream('mt19937ar','Seed',71000301);
 payload=int8(randi(stream,[0 1],r.PayloadBits,1));
 llr=14*(1-2*double(r.Codec.encode(payload)));
 for mode=["Normalized min-sum" "Layered belief propagation"]
  [bits,failed,iterations]=r.Codec.decode(llr,50,char(mode));
  assert(~failed && isequal(payload,bits) && all(iterations>0,'all'));
 end
 assert(Qm==r.Codec.Qm);
 if numRB==66
  % Exercise the experimental Qm=9 permutation with all four RVs too.
  combined=[];
  for rv=[0 2 3 1]
   coded=r.Codec.encode(payload,rv);
   [bits,failed,~,combined]=r.Codec.decode(8*(1-2*double(coded)),50, ...
       'Layered belief propagation',rv,combined);
  end
  assert(~failed && isequal(payload,bits),'Qm=9 incremental redundancy failed');
 end
 % Exercise the actual interleaver, QAM mapper and soft demapper, not just LDPC.
 coded=r.Codec.encode(payload);
 symbols=qammod(coded(r.BitOrder),cfg.M,'InputType','bit','UnitAveragePower',true);
 for snr=[30 40]
  noise=10^(-snr/10);
  received=symbols+sqrt(noise/2)*(randn(stream,size(symbols))+1i*randn(stream,size(symbols)));
  demapped=qamdemod(received,cfg.M,'OutputType','approxllr','UnitAveragePower',true,'NoiseVariance',noise);
  restored=zeros(r.CodedBits,1);restored(r.BitOrder)=max(-50,min(50,demapped));
  [bits,failed]=r.Codec.decode(restored,50,'Layered belief propagation');
  assert(~failed && isequal(payload,bits),'512-QAM/LDPC AWGN round-trip failed');
 end
 if numRB==66
  % A residual common phase must not be mistaken for additional ICI noise.
  [grid,payload]=muPayload(cfg,r,71000302);
  h=complex(zeros(r.K,r.L,16,4));
  for user=1:4
   for layer=1:4,h(:,:,(user-1)*4+layer,layer)=1;end
  end
  observed=sum(h.*reshape(grid,r.K,r.L,1,4),4)*exp(.37i);
  input=struct('Observed',observed,'Noise',1e-4*ones(1,16));
  [errors,failed,detail]=v2Decode(input,h,zeros(size(h)),payload,cfg,r);
  assert(all(errors==0 & ~failed),'Residual-phase receiver round-trip failed');
  assert(all(detail.MeanEffectiveVariance<1.01e-4), ...
   'Removed common phase must not inflate the LLR noise variance');
  [capturedErrors,capturedFailed,capturedDetail,primary]=v2Decode(input,h,zeros(size(h)),payload,cfg,r);
  assert(isequal(errors,capturedErrors) && isequal(failed,capturedFailed) && isequal(detail,capturedDetail), ...
   'Capturing diagnostics changed receiver behavior');
  assert(max(abs(primary.PTRSAmplitude-1),[],'all')<1e-10);
  input.Observed=.82*observed;
  [~,~,~,primary]=v2Decode(input,h,zeros(size(h)),payload,cfg,r);
  assert(max(abs(primary.PTRSAmplitude-.82),[],'all')<1e-10, ...
   'Known-reference amplitude diagnostic is incorrectly normalized');
 end
end
% The optional coherent precoder must preserve power, CDM pairs and time support.
cfg=v3Config(32,12);cfg.BeamWidth=24;cfg.PrecoderTimeMode='slot';r=v3Resources(cfg);
scene=struct('Config',cfg,'Front',muFrontEnd(32,8,.7,.3));
stream=RandStream('mt19937ar','Seed',71000303);
h=complex(randn(stream,r.K,r.L,8,4),randn(stream,r.K,r.L,8,4));
for scheme=["MR" "ZF" "PZF"]
 [weights,diagnostic]=v3Weights(h,.01*ones(size(h)),scheme,scene,r);
 assert(diagnostic.MaximumPowerError<1e-6);
 reference=repmat(weights(:,:,:,6),1,1,1,7);
 assert(max(abs(weights(:,:,:,6:12)-reference),[],'all')<3e-7, ...
  'Coherent precoder changed between reference and data symbols');
 for group=1:numel(r.DMRSGroups)
  pairs=r.DMRSGroups{group}.Pairs;
  assert(max(abs(weights(:,:,pairs(1,:),6)-weights(:,:,pairs(2,:),6)),[],'all')<3e-7, ...
   'CDM pilot pair uses inconsistent precoders');
 end
end
% Disjoint pilots must identify each port even when H varies between tones.
cfg=v3Config(32,12);baseline=v3Resources(cfg);cfg.OrthogonalDMRS=true;r=v3Resources(cfg);
assert(isequal(r.DataIndices,baseline.DataIndices) && r.CodedBits==baseline.CodedBits);
assert(max(abs(sum(abs(r.ReferenceGrid).^2,3)-sum(abs(baseline.ReferenceGrid).^2,3)),[],'all')<1e-10);
truth=complex(randn(stream,r.K,r.L,16,4),randn(stream,r.K,r.L,16,4));
observed=sum(truth.*reshape(r.ReferenceGrid,r.K,r.L,1,4),4);
samples=v3PilotObservations(observed,r);
for user=1:4
 mask=repmat(abs(r.ReferenceGrid(:,:,user))>0,1,1,16);
 actual=samples(:,:,:,user);expected=truth(:,:,:,user);
 assert(max(abs(actual(mask)-expected(mask)))<1e-11, ...
  'Orthogonal pilots failed to separate frequency-varying user channels');
end
fprintf('V4_RECEIVER_SELF_TEST_PASS: rate 3/4, 66/264 RB, QAM and AWGN round trips\n');
end
