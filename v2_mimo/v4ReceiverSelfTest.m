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
