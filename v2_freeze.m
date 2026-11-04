function v2_freeze
addpath('mu_mimo','v2_mimo');folder='v2_frozen';if ~isfolder(folder),mkdir(folder);end
model=v2Assemble('v2_tx_training/v2_tx.mat','rx_models');
% Exercise the actual 66-RB inference and decoding path before payload selection.
for antennas=[32 64]
 cfg=v2Config(antennas);r=muResources(cfg);
 result=v2Benchmark(cfg,r,model,1,[0 10 20],8900000+antennas*1000);
 v2Export(result,model,fullfile(folder,'smoke_BS'+string(antennas)));
end
model=v2SelectPayload(model,folder,8);
save(fullfile(folder,'v2_model.mat'),'model','-v7.3');
summary=struct('Version',model.Version,'BlendTX',model.BlendTX,'Selection',model.Selection, ...
 'TrainingChannelsTX',model.TrainingFrames,'ValidationChannelsTX',model.ValidationFrames, ...
 'TrainingChannelsRX',struct,'ValidationChannelsRX',struct,'BlendRX',struct);
for scheme=["MR" "ZF" "PZF"]
 s=model.Receivers.(scheme);summary.TrainingChannelsRX.(scheme)=s.TrainingChannels;
 summary.ValidationChannelsRX.(scheme)=s.ValidationChannels;summary.BlendRX.(scheme)=s.BlendRX;
end
fid=fopen(fullfile(folder,'v2_model_summary.json'),'w');fprintf(fid,'%s',jsonencode(summary));fclose(fid);
fprintf('V2_MODEL_SUMMARY_BEGIN\n%s\nV2_MODEL_SUMMARY_END\n',jsonencode(summary));
end
