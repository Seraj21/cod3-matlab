function v3_freeze
addpath(pwd,'mu_mimo','v2_mimo');folder='v3_frozen';if ~isfolder(folder),mkdir(folder);end
model=v3Assemble('v2_frozen/v2_model.mat','rx_models');model=v3SelectPayload(model,folder);
for bs=[32 64]
 cfg=v3Config(bs);r=v3Resources(cfg);v3Benchmark(cfg,r,model,1,[0 10 20],19500000+bs*1000,fullfile(folder,'smoke_BS'+string(bs)));
end
save(fullfile(folder,'v3_model.mat'),'model','-v7.3');
summary=struct('Version',model.Version,'BlendTX',model.BlendTX,'Selection',model.Selection, ...
 'TrainingChannelsTX',model.TrainingFrames,'ValidationChannelsTX',model.ValidationFrames, ...
 'TrainingChannelsRX',struct,'ValidationChannelsRX',struct,'BlendRX',struct, ...
 'PilotMode',"DMRS+PTRS",'ReferencesPerLocalChannel',22,'TargetCodeRate',.125, ...
 'PZFMode',model.Signature.PZFMode,'AttentionGatesPerReceiver',2);
for scheme=["MR" "ZF" "PZF"]
 s=model.Receivers.(scheme);summary.TrainingChannelsRX.(scheme)=s.TrainingChannels;
 summary.ValidationChannelsRX.(scheme)=s.ValidationChannels;summary.BlendRX.(scheme)=s.BlendRX;
end
fid=fopen(fullfile(folder,'v3_model_summary.json'),'w');fprintf(fid,'%s',jsonencode(summary));fclose(fid);
fprintf('V3_FROZEN_SUMMARY_BEGIN\n%s\nV3_FROZEN_SUMMARY_END\n',jsonencode(summary));
end
