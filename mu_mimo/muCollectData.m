function data = muCollectData(base,r,trainingFrames,validationFrames,folder)
% Splits are by independent channel realizations, before patch extraction.
trainTX = cell(trainingFrames,1);
trainRX = cell(2*trainingFrames,1);
validTX = cell(validationFrames,1);
validRX = cell(2*validationFrames,1);
statsTX = muPrior("init",r.UplinkLayout);
statsRX = muPrior("init",r.DownlinkLayout);
stream = RandStream('mt19937ar','Seed',910011);
logRows = zeros(trainingFrames+validationFrames,9);
schemes = ["MR" "ZF" "PZF"];
for index = 1:trainingFrames+validationFrames
    training = index<=trainingFrames;
    frame = index;
    if ~training,frame=index-trainingFrames;end
    seed = 1000000+frame;
    if ~training,seed=2000000+frame;end
    cfg = base;
    cfg.Antennas = 32+32*mod(frame,2);
    if mod(frame,4)~=0
        cfg.DelaySpread = 10^(-7+rand(stream));
        speeds = [0 30 60 90 120];
        cfg.Speed = speeds(randi(stream,numel(speeds)));
        cfg.VerticalCorrelation = .4+.5*rand(stream);
        cfg.HorizontalCorrelation = .1+.4*rand(stream);
        cfg.UECorrelation = .5*rand(stream);
        cfg.SoundingSNR = 5+20*rand(stream);
    end
    scene = muSounding(cfg,r,seed);
    bands = sort(unique(mod(frame-1+[0 5 10 15 20 25],r.G)+1));
    selected = reshape(((bands(:)-1)*24+(1:24)).',[],1);
    recTX = record(scene.ULSamples,scene.ULTruth,scene.ULNoise,selected,seed,cfg.Antennas);
    if training
        statsTX = muPrior("accumulate",statsTX,scene.ULCleanSamples,scene.ULTruth,r.UplinkLayout);
        trainTX{frame} = recTX;
    else
        validTX{frame} = recTX;
    end
    hls = muLS(scene.ULSamples,r.UplinkLayout);
    alpha = [0 .5 1];
    alpha = alpha(mod(frame-1,3)+1);
    h = (1-alpha)*scene.ULTruth+alpha*hls;
    scheme = schemes(mod(frame-1,numel(schemes))+1);
    weights = muWeights(h,scheme,scene,r);
    [grid,~] = muPayload(cfg,r,muSeed(seed,70));
    packet = muDownlink(scene,r,grid,weights);
    snr = 20*rand(stream);
    points = [snr mod(snr+10,20)];
    for noiseIndex = 1:2
        input = muReceive(packet,scene,r,points(noiseIndex),false,true);
        recRX = record(input.Samples,input.Truth,input.Noise,selected,seed,cfg.Antennas);
        if training
            statsRX = muPrior("accumulate",statsRX,input.CleanSamples,input.Truth,r.DownlinkLayout);
            trainRX{2*(frame-1)+noiseIndex} = recRX;
        else
            validRX{2*(frame-1)+noiseIndex} = recRX;
        end
    end
    logRows(index,:) = [training frame seed cfg.Antennas cfg.DelaySpread*1e9 cfg.Speed cfg.SoundingSNR points];
    if mod(index,16)==0 || index==trainingFrames+validationFrames
        fprintf('Collected %d/%d independent channel realizations\n',index,trainingFrames+validationFrames);
    end
    clear scene packet input
end
data = struct('TrainTX',{trainTX},'TrainRX',{trainRX},'ValidTX',{validTX},'ValidRX',{validRX}, ...
    'PriorsTX',muPrior("finalize",statsTX),'PriorsRX',muPrior("finalize",statsRX));
writetable(array2table(logRows,'VariableNames',{'Training','Frame','ChannelSeed','BSAntennas', ...
    'RMSDelay_ns','Speed_kmh','SoundingSNR_dB','DataSNR1_dB','DataSNR2_dB'}),fullfile(folder,'dataset.csv'));
end
function item = record(samples,truth,noise,selected,seed,antennas)
item = struct('Samples',single(samples(selected,:,:,:)),'Truth',single(truth(selected,:,:,:)), ...
    'Noise',noise,'Seed',seed,'Antennas',antennas);
end
