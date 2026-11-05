function run_v4_frozen_pzf(antennas,batch,stage)
% All selection occurs before the test-seed sweep.
if stage=="validation"
 choice=jsondecode(fileread('receiver_choice/receiver.json'));
 modes=string(choice.retry);points=0:2:20;seed=74000000;
 methods=["LS" "LMMSE" "V3 CNN" "Matched LMMSE" "Matched CNN" "Aligned CNN"];
else
 assert(stage=="test");
 choice=jsondecode(fileread('frozen_choice/pzf_selection.json'));
 modes=unique(["None" string(choice.common_receiver.retry)],'stable');
 methods=unique(["LS" "LMMSE" "Matched LMMSE" "V3 CNN" string(choice.cnn) "Perfect CSI"],'stable');
 points=[0:.5:10 11:20];seed=90000000;
end
run_v4_pzf(antennas,4,'v4_measurement',points,seed+batch*100000, ...
 'v3_refined/v3_model.mat',modes,methods, ...
 'models/v4-pzf-matched/pzf_receiver.mat','models/v4-pzf-aligned/pzf_receiver.mat');
end
