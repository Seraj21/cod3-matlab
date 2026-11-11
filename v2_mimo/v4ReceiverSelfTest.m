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
end
fprintf('V4_RECEIVER_SELF_TEST_PASS: rate 3/4, 66/264 RB, QAM and AWGN round trips\n');
end
