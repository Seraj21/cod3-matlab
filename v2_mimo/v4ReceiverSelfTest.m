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
cfg=v3Config(32);r=v3Resources(cfg);
stream=RandStream('mt19937ar','Seed',71000301);
payload=int8(randi(stream,[0 1],r.PayloadBits,1));
llr=14*(1-2*double(r.Codec.encode(payload)));
for mode=["Normalized min-sum" "Layered belief propagation"]
 [bits,failed,iterations]=r.Codec.decode(llr,50,char(mode));
 assert(~failed && isequal(payload,bits) && all(iterations>0,'all'));
end
assert(Qm==r.Codec.Qm);
end
