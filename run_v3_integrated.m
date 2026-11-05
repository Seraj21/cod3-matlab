function run_v3_integrated(precoder,antennas,batch,stage)
% V3 precoders and the V4 receiver share the same validation and test design.
precoder=string(precoder);stage=string(stage);
assert(ismember(precoder,["MR" "ZF" "PZF"]));
assert(ismember(antennas,[32 64]));
assert(ismember(batch,0:7));
methods=["LS on common packet" "LMMSE" "Aligned LMMSE"];
if stage=="validation"
 assert(batch<4);
 modes=["APP50" "CSI-APP50"];
 methods=[methods "V3 CNN" "Aligned CNN"];
 snr=[0 4 6 8 10 12 16 20];seed=76000000;
else
 assert(stage=="test");
 choice=jsondecode(fileread('integrated_selection/selection.json'));
 assert(choice.Version==1 && ismember(string(choice.Retry),["APP50" "CSI-APP50"]));
 candidate=string(choice.CNN.(precoder));
 assert(ismember(candidate,["V3 CNN" "Aligned CNN"]));
 modes=["None" string(choice.Retry)];
 methods=[methods candidate "Perfect CSIR"];
 snr=[0:.5:10 11:20];seed=92000000;
end
options=struct('Precoder',precoder,'RetryModes',modes,'Methods',methods, ...
 'AlignedModel',fullfile('integrated_receiver',lower(precoder)+"_receiver.mat"));
run_v3_mimo(antennas,4,'integrated_measurement',snr,seed+batch*100000, ...
 'v3_refined/v3_model.mat',[],options);
end
