function tests=testV2Core
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'mu_mimo'),fullfile(root,'v2_mimo'));
end
function testBatchedPrecoder(testCase)
cfg=v2Config(32,12);cfg.BeamWidth=24;cfg.AlignPhase=false;r=muResources(cfg);
scene=struct('Config',cfg,'Front',muFrontEnd(32,8,.7,.3));
stream=RandStream('mt19937ar','Seed',731);
H=complex(randn(stream,4,8),randn(stream,4,8));
h=repmat(reshape(H.',1,1,8,4),r.K,r.L,1,1);
for scheme=["MR" "ZF" "PZF"]
 [weights,diagnostic]=v2Weights(h,scheme,scene,r);
 reference=muPrecoder(H,scheme,scene.Front,4,2);
 verifyEqual(testCase,double(weights(:,:,1,6)),reference,'AbsTol',3e-7);
 verifyLessThan(testCase,diagnostic.MaximumPowerError,1e-8);
 if scheme=="PZF"
  verifyLessThanOrEqual(testCase,diagnostic.MeanFinalObjective,diagnostic.MeanInitialObjective+1e-9);
 end
end
end
function testPhaseInvariantGateFeatures(testCase)
cfg=v2Config(32,12);r=muResources(cfg);
stream=RandStream('mt19937ar','Seed',732);
x=complex(randn(stream,r.K,12,8,4),randn(stream,r.K,12,8,4));
phase=exp(1i*2*pi*rand(stream,1,1,8,4));
a=v2GateFeatures(x,.03,r.UplinkLayout);
b=v2GateFeatures(x.*phase,.03,r.UplinkLayout);
verifyEqual(testCase,a,b,'AbsTol',3e-6);
net=v2GateNetwork(8,4);p=extractdata(predict(net,dlarray(a,'SSCB')));
verifyEqual(testCase,p,.2*ones(5,r.G,'single'),'AbsTol',1e-7);
end
function testAffinePredictionOnIndependentSamples(testCase)
cfg=v2Config(32,12);r=muResources(cfg);layouts=r.UplinkLayout;
stats=v2Prior("init",layouts,0);stream=RandStream('mt19937ar','Seed',733);
for i=1:80
 truth=repmat(reshape(2+2i+(randn(stream,r.G,8,4)+1i*randn(stream,r.G,8,4))/sqrt(2),1,r.G,1,8,4),24,1,12,1,1);
 truth=reshape(truth,r.K,12,8,4);
 stats=v2Prior("accumulate",stats,truth,truth,layouts);
end
models=v2Prior("finalize",stats);
truth=repmat(reshape(2+2i+(randn(stream,r.G,8,4)+1i*randn(stream,r.G,8,4))/sqrt(2),1,r.G,1,8,4),24,1,12,1,1);
truth=reshape(truth,r.K,12,8,4);
noise=(randn(stream,size(truth))+1i*randn(stream,size(truth)))/sqrt(2)*.2;
state=v2CEBase(truth+noise,.04,layouts,models);
verifyLessThan(testCase,mean(abs(state.Wiener-truth).^2,'all'),mean(abs(state.LS-truth).^2,'all'));
verifyGreaterThanOrEqual(testCase,min(state.VarianceWiener,[],'all'),0);
end
function testRealParameterPacking(testCase)
cfg=v2Config(32,12);r=muResources(cfg);layouts=r.DownlinkLayout;
stream=RandStream('mt19937ar','Seed',734);
h=randn(stream,r.K,12,4,4);
meta=struct('W',24,'L',12,'G',r.G,'GroupSize',4,'U',4,'Coefficients',16,'K',r.K);
p=v2ParameterPatches(h,meta);back=v2ParameterRestore(p,meta);
verifyEqual(testCase,double(back),h,'AbsTol',3e-7);
end
function testCRCOnlyCombinerSelection(testCase)
cfg=v2Config(32,12);r=muResources(cfg);[grid,payload]=muPayload(cfg,r,735);
stream=RandStream('mt19937ar','Seed',736);
trueH=complex(zeros(16,4));estimated=trueH;
for u=1:4
 rows=(u-1)*4+(1:4);trueH(rows(1),u)=1;estimated(rows(1),u)=1;
 trueH(rows(2:4),:)=complex(randn(stream,3,4),randn(stream,3,4));
 estimated(rows(2:4),u)=10;
end
truth=repmat(reshape(trueH,1,1,16,4),r.K,12,1,1);
h=repmat(reshape(trueH,1,1,16,4),r.K,12,1,1);
wrong=repmat(reshape(estimated,1,1,16,4),r.K,12,1,1);
dataMask=repmat(r.DataMask,1,1,16,4);h(dataMask)=wrong(dataMask);
observed=sum(truth.*reshape(grid,r.K,12,1,4),4);
input=struct('Observed',observed,'Noise',1e-9*ones(1,16));
[e,f,d]=v2Decode(input,h,zeros(size(h)),payload,cfg,r);
verifyEqual(testCase,e,zeros(1,4));verifyFalse(testCase,any(f));
verifyTrue(testCase,all(d.SRSBeamAccepted));
incorrect=payload;incorrect(1,:)=1-incorrect(1,:);
[e2,f2,d2]=v2Decode(input,h,zeros(size(h)),incorrect,cfg,r);
verifyEqual(testCase,e2,ones(1,4));verifyEqual(testCase,f2,f);
verifyEqual(testCase,d2.SRSBeamAccepted,d.SRSBeamAccepted);
end
function testCNNUncertaintyLossGradient(testCase)
net=v2RXNetwork(584,16);
X=dlarray(randn(6,12,584,2,'single'),'SSCB');
T=dlarray(cat(3,.1*randn(6,12,128,2,'single'),-3*ones(6,12,64,2,'single')),'SSCB');
[loss,gradients]=dlfeval(@gradientLoss,net,X,T);
verifyTrue(testCase,isfinite(extractdata(loss)));
norms=cellfun(@(g)sum(abs(extractdata(g)).^2,'all'),gradients.Value);
verifyGreaterThan(testCase,sum(norms),0);verifyTrue(testCase,all(isfinite(norms)));
end
function [loss,gradients]=gradientLoss(net,X,T)
prediction=forward(net,X);loss=v2RXLoss(prediction,T);
gradients=dlgradient(loss,net.Learnables);
end
