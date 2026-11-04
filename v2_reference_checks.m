function v2_reference_checks
% Independent contracts from MathWorks NR and Sionna's published equations.
addpath('mu_mimo','v2_mimo');folder='v2_reference';if ~isfolder(folder),mkdir(folder);end
stream=RandStream('mt19937ar','Seed',750001);
modulations=["QPSK" "16QAM" "64QAM" "256QAM"];
for A=[2048 11880]
 for index=1:4
  qm=2*index;E=qm*floor(2*A/qm);codec=ExperimentalLDPC(A,E,qm,.5);
  payload=int8(randi(stream,[0 1],A,1));info=nrDLSCHInfo(A,.5);
  mother=nrLDPCEncode(nrCodeBlockSegmentLDPC(nrCRCEncode(payload,info.CRC),info.BGN),info.BGN);
  expected=nrRateMatchLDPC(mother,E,0,modulations(index),1);
  assert(isequal(codec.encode(payload),expected),'NR rate-matching reference mismatch');
 end
end
labels=dec2bin(0:511,9)-'0';constellation=qammod((0:511).',512,'UnitAveragePower',true);
assert(abs(mean(abs(constellation).^2)-1)<1e-12);
bits=reshape(labels.',[],1);mapped=qammod(bits,512,'InputType','bit','UnitAveragePower',true);
assert(max(abs(mapped-constellation))<1e-12);
noise=.07;y=constellation(randi(stream,512,1024,1))+sqrt(noise/2)* ...
 (randn(stream,1024,1)+1i*randn(stream,1024,1));
distance=abs(y-constellation.').^2;direct=zeros(9,numel(y));
for b=1:9
 direct(b,:)=(min(distance(:,labels(:,b)==1),[],2)-min(distance(:,labels(:,b)==0),[],2)).'/noise;
end
native=qamdemod(y,512,'OutputType','approxllr','UnitAveragePower',true,'NoiseVariance',noise);
llrDifference=max(abs(native-direct(:)));
assert(llrDifference<1e-9,'512-QAM max-log bit labeling or variance convention mismatch');
% The Sionna reference uses diag(GH)^(-1)-I. Compare a separate whitening solve.
noiseDifference=0;
for trial=1:40
 H=(randn(stream,4,4)+1i*randn(stream,4,4))/sqrt(2);
 B=(randn(stream,4,4)+1i*randn(stream,4,4))/sqrt(2);
 S=B*B'+.01*eye(4);
 reference=(H'*(S\H)+eye(4))\(H'/S);
 gain=diag(reference*H);expected=real(1./gain)-1;
 W=(H*H'+S)\H;actual=zeros(4,1);
 for user=1:4
  w=W(:,user);alpha=w'*H(:,user);response=w'*H;response(user)=0;
  actual(user)=(sum(abs(response).^2)+real(w'*S*w))/abs(alpha)^2;
 end
 noiseDifference=max(noiseDifference,max(abs(expected-actual)));
end
assert(noiseDifference<1e-9,'LMMSE gain normalization or effective variance mismatch');
cfg=v2Config(64);r=muResources(cfg);points=[0 4 8 12 16 20];errors=zeros(1,6);failed=errors;
for frame=1:8
 payload=int8(randi(stream,[0 1],r.PayloadBits,1));coded=r.Codec.encode(payload);
 x=qammod(coded(r.BitOrder),512,'InputType','bit','UnitAveragePower',true);
 base=(randn(stream,size(x))+1i*randn(stream,size(x)))/sqrt(2);
 for s=1:6
  variance=10^(-points(s)/10);y=x+sqrt(variance)*base;
  llr=qamdemod(y,512,'OutputType','approxllr','UnitAveragePower',true,'NoiseVariance',variance);
  restored=zeros(r.CodedBits,1);restored(r.BitOrder)=llr;
  [decoded,bad]=r.Codec.decode(max(-50,min(50,restored)));
  e=nnz(decoded~=payload);errors(s)=errors(s)+e;failed(s)=failed(s)+(bad || e>0);
 end
end
assert(errors(end)<.03*8*r.PayloadBits && failed(end)<8 && errors(1)>.15*8*r.PayloadBits, ...
 'The isolated coded-AWGN control does not show a decoding transition');
T=table(points.',errors.',repmat(8*r.PayloadBits,6,1),failed.', ...
 'VariableNames',{'SNR_dB','PayloadBitErrors','PayloadBits','FailedTransportBlocks'});
writetable(T,fullfile(folder,'awgn_512_ldpc.csv'));
summary=struct('NRRateMatchingCases',8,'Max512QAMLLRDifference',llrDifference, ...
 'MaxLMMSEVarianceDifference',noiseDifference,'AwgnCounts',table2struct(T));
fid=fopen(fullfile(folder,'reference_audit.json'),'w');fprintf(fid,'%s',jsonencode(summary));fclose(fid);
fprintf('V2_REFERENCE_CHECKS_PASS\n%s\nV2_REFERENCE_CHECKS_END\n',jsonencode(summary));
end
