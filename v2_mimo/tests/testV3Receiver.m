function tests=testV3Receiver
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(root,fullfile(root,'mu_mimo'),fullfile(root,'v2_mimo'));
end
function testExistingPilotResourcesAndExactObservations(testCase)
cfg=v2Config(32,12);cfg.Rate=.125;old=muResources(cfg);r=v3Resources(cfg);
verifyEqual(testCase,r.ReferenceGrid,old.ReferenceGrid);verifyEqual(testCase,r.DataIndices,old.DataIndices);
verifyEqual(testCase,r.CodedBits,old.CodedBits);verifyEqual(testCase,r.SRS,old.SRS);
stream=RandStream('mt19937ar','Seed',773);H=complex(randn(stream,16,4),randn(stream,16,4));
truth=repmat(reshape(H,1,1,16,4),r.K,r.L,1,1);
observed=sum(truth.*reshape(r.ReferenceGrid,r.K,r.L,1,4),4);samples=v3PilotObservations(observed,r);
for user=1:4
 p=r.DownlinkLayout{user};verifyEqual(testCase,p.Count,22);verifyEqual(testCase,p.NoiseGain,.25,'AbsTol',1e-12);
 verifyEqual(testCase,p.Linear(:,p.Observed),eye(22),'AbsTol',1e-10);
 flat=reshape(samples(:,:,:,user),r.K*r.L,16);mask=abs(r.PTRS(:,:,user))>0;
 verifyEqual(testCase,flat(find(mask),:),repmat(H(:,user).',nnz(mask),1),'AbsTol',1e-12);
 reconstructed=muLS(samples,r.DownlinkLayout);
 verifyEqual(testCase,reconstructed(:,6:12,:,:),truth(:,6:12,:,:),'AbsTol',1e-10);
end
end
function testAttentionMeanAndVarianceIdentityWithoutTruthAccess(testCase)
r=v3Resources(v2Config(32,12));stream=RandStream('mt19937ar','Seed',774);
layouts=cell(8,1);at=0;
for role=["Own" "Other"]
 for u=1:4
  at=at+1;layouts{at}=r.DownlinkLayout{u};layouts{at}.Label=role+"_"+layouts{at}.Label;
 end
end
stats=v2Prior("init",layouts,4);input=fixture(r,stream);
for user=1:4
 [samples,truth,p]=v2RXCanonical(input,r,user);stats=v2Prior("accumulate",stats,samples,truth,p);
end
warm=v2RXNetwork(584,16);net=v3RXNetwork(584,16,warm);
model=struct('PriorsRX',v2Prior("finalize",stats),'NetworkRX',net,'BlendRX',1);
for nrb=[12 66]
 r=v3Resources(v2Config(32,nrb));input=fixture(r,stream);
 [a,av,ay]=v2EstimateRX(input,model,r,"LMMSE");[b,bv,by]=v2EstimateRX(input,model,r,"CNN");
 verifyEqual(testCase,b,a,'AbsTol',1e-9);verifyEqual(testCase,bv,av,'AbsTol',1e-8);verifyEqual(testCase,by,ay);
 changed=input;changed.Truth=100*input.Truth+100i;[c,cv]=v2EstimateRX(changed,model,r,"CNN");
 verifyEqual(testCase,c,b);verifyEqual(testCase,cv,bv);
end
end
function testAttentionGradient(testCase)
net=v3RXNetwork(584,16);values=net.Learnables;
i=find(string(values.Layer)=="mean_and_log_variance" & string(values.Parameter)=="Weights");
values.Value{i}=dlarray(.001*randn(size(values.Value{i}),'single'));net=dlupdate(@(~,value)value,net,values);
X=dlarray(randn(6,12,584,2,'single'),'SSCB');T=dlarray(cat(3,.1*randn(6,12,128,2,'single'),-3*ones(6,12,64,2,'single')),'SSCB');
[loss,gradients]=dlfeval(@gradientLoss,net,X,T);verifyTrue(testCase,isfinite(extractdata(loss)));
norms=cellfun(@(g)sum(abs(extractdata(g)).^2,'all'),gradients.Value);
attention=contains(string(gradients.Layer),'attention_expand');verifyGreaterThan(testCase,sum(norms(attention)),0);
verifyTrue(testCase,all(isfinite(norms)));
end
function testResidualLSKeepsConsistentLegacyInterpolation(testCase)
r=v3Resources(v2Config(32,12));layouts=v3ResidualLSLayouts(r);
stream=RandStream('mt19937ar','Seed',775);samples=complex(zeros(r.K,r.L,2,4));
for user=1:4
 p=r.DMRSLegacyLayouts{user};mask=repmat(p.Mask,r.K/p.W,1,2);
 a=complex(randn(stream,r.K,r.L,2),randn(stream,r.K,r.L,2));a(~mask)=0;
 samples(:,:,:,user)=a;
end
baseline=muLS(samples,r.DMRSLegacyLayouts);
for user=1:4
 p=r.DownlinkLayout{user};old=r.DMRSLegacyLayouts{user};
 mask=repmat(p.Mask & ~old.Mask,r.K/p.W,1,2);
 a=samples(:,:,:,user);b=baseline(:,:,:,user);a(mask)=b(mask);samples(:,:,:,user)=a;
 verifyEqual(testCase,p.Count,22);
 verifyEqual(testCase,layouts{user}.Linear(:,p.Observed),eye(22),'AbsTol',1e-10);
end
reconstructed=muLS(samples,layouts);
verifyEqual(testCase,reconstructed,baseline,'AbsTol',1e-10);
end
function input=fixture(r,stream)
y=complex(randn(stream,r.K,r.L,16),randn(stream,r.K,r.L,16));
truth=complex(randn(stream,r.K,r.L,16,4),randn(stream,r.K,r.L,16,4));
input=struct('Observed',y,'Samples',v3PilotObservations(y,r),'Truth',truth,'Noise',.05*ones(1,16));
end
function [loss,gradients]=gradientLoss(net,X,T)
p=forward(net,X);loss=v2RXLoss(p,T);gradients=dlgradient(loss,net.Learnables);
end
