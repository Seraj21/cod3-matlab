function run_v3_integrated(precoder,antennas,batch,stage,numRB)
% V3 precoders and the V4 receiver share the same validation and test design.
addpath('mu_mimo','v2_mimo');
precoder=string(precoder);stage=string(stage);
if nargin<5,numRB=264;end
assert(ismember(precoder,["MR" "ZF" "PZF"]));
assert(ismember(antennas,[32 64]));
assert(ismember(batch,0:7));
assert(ismember(numRB,[66 264]));
frames=4;validationSeed=76000000;testSeed=92000000;
validationSNR=[0 4 6 8 10 12 16 20 25 30 35 40];
if numRB==264
 frames=2;validationSeed=82000000;testSeed=94000000;
 validationSNR=[0 6 10 14 20 25 30 35 40];
end
methods=["LS on common packet" "LMMSE" "Aligned LMMSE"];
cfg=v3Config(antennas,numRB,"mobile");
if stage=="validation"
 assert(batch<4);
 modes=["APP50" "CSI-APP50"];
 methods=[methods "V3 CNN" "Aligned CNN" "Perfect CSIR"];
 snr=validationSNR;seed=validationSeed;
else
 assert(stage=="test");
 choice=jsondecode(fileread('integrated_selection/selection.json'));
 assert(choice.Version==1 && ismember(string(choice.Retry),["APP50" "CSI-APP50"]));
 assert(choice.CodeRate==cfg.Rate && choice.MaxSNR_dB==40, ...
  'Repeat validation before testing the new code rate and SNR range');
 assert(isfield(choice,'OperatingProfile') && string(choice.OperatingProfile)==string(cfg.OperatingProfile), ...
  'Repeat validation before changing the channel operating profile');
 assert(isfield(choice,'ChannelConfig') && isfield(choice,'PrecoderConfig'), ...
  'Repeat validation with explicit channel and precoder configuration');
 for group=["ChannelConfig" "PrecoderConfig"]
  for name=string(fieldnames(choice.(group))).'
   assert(isequal(choice.(group).(name),cfg.(name)), ...
    'Validation/deployment configuration mismatch: %s',name);
  end
 end
 if isfield(choice,'NumRB'),assert(choice.NumRB==numRB);else,assert(numRB==66);end
 candidate=string(choice.CNN.(precoder));
 assert(ismember(candidate,["V3 CNN" "Aligned CNN"]));
 modes=["None" string(choice.Retry)];
 methods=[methods candidate "Perfect CSIR"];
 snr=[0:.5:10 11:40];seed=testSeed;
end
options=struct('Precoder',precoder,'RetryModes',modes,'Methods',methods,'NumRB',numRB, ...
 'AlignedModel',fullfile('integrated_receiver',lower(precoder)+"_receiver.mat"));
run_v3_mimo(antennas,frames,'integrated_measurement',snr,seed+batch*100000, ...
 'v3_refined/v3_model.mat',[],options);
end
