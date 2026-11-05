function run_v3_mimo(antennas,frames,folder,snrDb,seedOffset,modelFile,methodIndices,receiverOptions)
if nargin<4,snrDb=[0:.5:10 11:20];end
if nargin<5,seedOffset=50000000;end
if nargin<6
 modelFile='v3_frozen/v3_model.mat';
 if isfile('v3_refined/v3_model.mat'),modelFile='v3_refined/v3_model.mat';end
end
if nargin<7,methodIndices=1:4;end
if nargin>=8 && ~isempty(receiverOptions)
 numRB=66;if isfield(receiverOptions,'NumRB'),numRB=receiverOptions.NumRB;end
 run_v4_pzf(antennas,frames,folder,snrDb,seedOffset,modelFile, ...
  receiverOptions.RetryModes,receiverOptions.Methods,"",receiverOptions.AlignedModel,receiverOptions.Precoder,numRB);
 return
end
addpath(pwd,'mu_mimo','v2_mimo');load(modelFile,'model');assert(model.Version==6);
cfg=v3Config(antennas);r=v3Resources(cfg);
expected=struct('Users',cfg.Users,'UERx',cfg.UERx,'RFChains',cfg.RFChains,'Qm',cfg.Qm,'M',cfg.M, ...
 'CSIWidth',cfg.PRGWidth,'BeamWidth',cfg.BeamWidth,'CPFraction',cfg.CPFraction,'PhaseBits',cfg.PhaseBits, ...
 'PilotMode',cfg.PilotMode,'PZFMode',cfg.PZFMode,'TargetCodeRate',cfg.Rate);
assert(isequal(model.Signature,expected),'V3 frozen model and geometry mismatch');
assert(all(diff(snrDb)>0) && min(snrDb)>=0 && max(snrDb)<=20);
v3Benchmark(cfg,r,model,frames,snrDb,seedOffset,folder,methodIndices);
end
