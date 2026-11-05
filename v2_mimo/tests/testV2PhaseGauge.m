function tests=testV2PhaseGauge
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(root,fullfile(root,'mu_mimo'),fullfile(root,'v2_mimo'));
end
function testPhaseReferencePreservesBeamPowersAndRelativePhaseGrid(testCase)
stream=RandStream('mt19937ar','Seed',772);
for width=[1 4]
 cfg=v2Config(32,12);cfg.BeamWidth=width;r=muResources(cfg);scene=struct('Config',cfg);
 H=complex(randn(stream,4,8),randn(stream,4,8));
 h=repmat(reshape(H.',1,1,8,4),r.K,r.L,1,1);
 base=v2Weights(h,"PZF",scene,r);
 for mode=["quantized" "continuous"]
  [w,d]=v2PhaseGauge(base,h,scene,r,mode);
  verifyLessThan(testCase,d.ResponsePowerRelativeError,1e-11);
  verifyLessThan(testCase,d.TotalPowerError,1e-11);
  response=H*double(w(:,:,1,6));desired=diag(response);
  limit=pi/16+1e-6;if mode=="continuous",limit=1e-6;end
  verifyLessThanOrEqual(testCase,max(abs(angle(desired))),limit);
  verifyEqual(testCase,abs(H*double(base(:,:,1,6))).^2,abs(response).^2,'AbsTol',3e-6);
 end
end
end
