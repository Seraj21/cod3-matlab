function run_v2_mimo(antennas,frames,folder,snrDb,seedOffset,modelFile,codeRate)
if nargin<4,snrDb=[0:.5:10 11:20];end
if nargin<5,seedOffset=20000000;end
if nargin<6,modelFile='v2_frozen/v2_model.mat';end
if nargin<7,codeRate=.5;end
addpath('mu_mimo','v2_mimo');
assert(all(diff(snrDb)>0) && min(snrDb)>=0 && max(snrDb)<=20);
loaded=load(modelFile,'model');model=loaded.model;assert(model.Version==5);
cfg=v2Config(antennas);cfg.Rate=codeRate;r=muResources(cfg);
expected=struct('Users',cfg.Users,'UERx',cfg.UERx,'RFChains',cfg.RFChains,'Qm',cfg.Qm, ...
 'M',cfg.M,'CSIWidth',cfg.PRGWidth,'BeamWidth',cfg.BeamWidth,'CPFraction',cfg.CPFraction, ...
 'PhaseBits',cfg.PhaseBits);
assert(isequal(model.Signature,expected),'Frozen CNN dimensions or pilot/precoder geometry mismatch');
result=v2Benchmark(cfg,r,model,frames,snrDb,seedOffset);
v2Export(result,model,folder);
end
