function tests=testV2RobustPhase
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'mu_mimo'),fullfile(root,'v2_mimo'));
end
function testSinglePrecisionWarmStartAndPhysicalConstraints(testCase)
stream=RandStream('mt19937ar','Seed',771);
for width=[1 4]
 cfg=v2Config(32,12);cfg.BeamWidth=width;r=muResources(cfg);scene=struct('Config',cfg);
 H=complex(randn(stream,4,8),randn(stream,4,8));
 h=repmat(reshape(H.',1,1,8,4),r.K,r.L,1,1);variance=.15*ones(size(h));
 old=v2Weights(h,"PZF",scene,r);verifyClass(testCase,old,'single');
 [w,d]=v2RobustPhaseWeights(h,variance,scene,r,old);
 active=double(w(:,:,:,cfg.DownlinkSymbols));amplitude=1/sqrt(32);step=2*pi/16;
 verifyLessThan(testCase,d.MaximumPowerError,1e-8);
 verifyLessThan(testCase,max(abs(sum(abs(active).^2,[1 2])-1),[],'all'),3e-7);
 verifyEqual(testCase,abs(active),amplitude*ones(size(active)),'AbsTol',3e-8);
 projected=amplitude*exp(1i*step*round(angle(active)/step));
 verifyEqual(testCase,active,projected,'AbsTol',3e-8);
 verifyLessThanOrEqual(testCase,d.MeanFinalObjective,d.MeanInitialObjective+1e-9);
end
end
