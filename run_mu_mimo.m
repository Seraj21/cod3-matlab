function run_mu_mimo(antennas,numPackets,outputFolder,snrDb,seedOffset,options)
arguments
    antennas (1,1) double {mustBeMember(antennas,[32 64])} = 32
    numPackets (1,1) double {mustBeInteger,mustBePositive} = 16
    outputFolder (1,1) string = "mu_results"
    snrDb (1,:) double = [0:.5:10 11:20]
    seedOffset (1,1) double = 10000000
    options.Mode (1,1) string {mustBeMember(options.Mode,["train" "evaluate"])} = "evaluate"
    options.ModelFile (1,1) string = "models/mu_cnn.mat"
    options.NumRB (1,1) double = 66
    options.TrainingFrames (1,1) double = 256
    options.ValidationFrames (1,1) double = 32
    options.BERValidationFrames (1,1) double = 16
    options.Epochs (1,1) double = 45
    options.Users (1,1) double = 4
    options.UERx (1,1) double = 4
    options.RFChains (1,1) double = 8
end
root=fileparts(mfilename('fullpath'));
addpath(fullfile(root,'mu_mimo'));
assert(all(isfinite(snrDb)) && all(diff(snrDb)>0) && min(snrDb)>=0 && max(snrDb)<=20);
if ~isfolder(outputFolder),mkdir(outputFolder);end
cfg=muConfig(antennas,NumRB=options.NumRB,Users=options.Users,UERx=options.UERx,RFChains=options.RFChains);
r=muResources(cfg);
if options.Mode=="train"
    parent=fileparts(options.ModelFile);
    if parent~="" && ~isfolder(parent),mkdir(parent);end
    data=muCollectData(cfg,r,options.TrainingFrames,options.ValidationFrames,outputFolder);
    model=muTrainNetworks(cfg,r,data,outputFolder,options.Epochs);
    clear data
    save(fullfile(outputFolder,'model_before_payload_selection.mat'),'model','-v7.3');
    model=muSelectBER(cfg,r,model,options.BERValidationFrames,outputFolder);
    save(options.ModelFile,'model','-v7.3');
    fprintf('Saved frozen joint CNN model: %s\n',options.ModelFile);
else
    assert(isfile(options.ModelFile),'A frozen MU model is required. Run Mode="train" first.');
    loaded=load(options.ModelFile,'model');model=loaded.model;
    signature=struct('Users',cfg.Users,'UERx',cfg.UERx,'RFChains',cfg.RFChains, ...
        'NumRB',cfg.NumRB,'M',cfg.M,'Qm',cfg.Qm,'PhaseBits',cfg.PhaseBits);
    assert(model.Version==4 && isequal(model.Signature,signature),'Cached model configuration mismatch.');
    result=muBenchmark(cfg,r,model,numPackets,snrDb,seedOffset);
    muExport(result,model,outputFolder);
end
end
