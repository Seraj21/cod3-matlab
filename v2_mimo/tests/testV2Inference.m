function tests=testV2Inference
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'mu_mimo'),fullfile(root,'v2_mimo'));
end
function testFullBandwidthInferenceAndNoTruthAccess(testCase)
cfg=v2Config(32,12);r=muResources(cfg);layouts=cell(4,1);at=0;
for role=["Own" "Other"]
 for u=[1 3]
  at=at+1;layouts{at}=r.DownlinkLayout{u};layouts{at}.Label=role+"_"+layouts{at}.Label;
 end
end
stats=v2Prior("init",layouts,4);stream=RandStream('mt19937ar','Seed',741);
input=fixture(r,stream);
for user=1:4
 [samples,truth,p]=v2RXCanonical(input,r,user);
 stats=v2Prior("accumulate",stats,samples,truth,p);
end
model=struct('PriorsRX',v2Prior("finalize",stats),'NetworkRX',v2RXNetwork(584,16),'BlendRX',1);
for numRB=[12 66]
 r=muResources(v2Config(32,numRB));input=fixture(r,stream);
 [a,av,ay]=v2EstimateRX(input,model,r,"LMMSE");
 [b,bv,by]=v2EstimateRX(input,model,r,"CNN");
 % A zero CNN output must preserve both the LMMSE mean and its uncertainty.
 verifyEqual(testCase,b,a,'AbsTol',1e-9);verifyEqual(testCase,bv,av,'AbsTol',1e-8);
 verifyEqual(testCase,by,ay);verifyTrue(testCase,all(isfinite(b),'all'));
 changed=input;changed.Truth=100*input.Truth+100i;
 [c,cv]=v2EstimateRX(changed,model,r,"CNN");
 verifyEqual(testCase,c,b);verifyEqual(testCase,cv,bv);
 [perfect,v,observed]=v2EstimateRX(input,model,r,"Perfect");
 verifyEqual(testCase,sum(abs(perfect).^2,'all'),sum(abs(input.Truth).^2,'all'),'RelTol',1e-12);
 verifyEqual(testCase,sum(abs(observed).^2,'all'),sum(abs(input.Observed).^2,'all'),'RelTol',1e-12);
 verifyEqual(testCase,v,zeros(size(perfect)));
end
end
function input=fixture(r,stream)
y=complex(randn(stream,r.K,r.L,16),randn(stream,r.K,r.L,16));
truth=complex(randn(stream,r.K,r.L,16,4),randn(stream,r.K,r.L,16,4));
input=struct('Observed',y,'Samples',muPilotObservations(y,r,"DL"), ...
 'Truth',truth,'Noise',.05*ones(1,16));
end
