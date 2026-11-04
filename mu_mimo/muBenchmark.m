function result = muBenchmark(cfg,r,model,numPackets,snrDb,seedOffset,schemes,methods)
arguments
    cfg
    r
    model
    numPackets (1,1) double
    snrDb (1,:) double
    seedOffset (1,1) double
    schemes (1,:) string = ["MR" "ZF" "PZF"]
    methods (1,:) double = 1:4
end
names = ["LS" "LMMSE" "CNN" "Perfect CSI"];
errors = zeros(numPackets,numel(snrDb),numel(schemes),numel(methods),cfg.Users);
failed = false(size(errors));
calibration = zeros(numPackets,numel(schemes),numel(methods),5);
for packetIndex = 1:numPackets
    seed = seedOffset+packetIndex;
    scene = muSounding(cfg,r,seed);
    [grid,payload] = muPayload(cfg,r,muSeed(seed,70));
    h = cell(4,1);
    [h{1},~,state] = muCE(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX,r,"LS");
    h{2}=state.Wiener;
    if any(methods==3)
        h{3}=muCE(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX,r,"CNN", ...
            model.NetworkTX,model.BlendTX*model.BlendMultiplier,cfg.RFChains);
    end
    h{4}=scene.ULTruth;
    for precoder = 1:numel(schemes)
        for estimator = 1:numel(methods)
            method = methods(estimator);
            [weights,d] = muWeights(h{method},schemes(precoder),scene,r);
            packet = muDownlink(scene,r,grid,weights);
            measured = zeros(numel(snrDb),1);
            for s = 1:numel(snrDb)
                input = muReceive(packet,scene,r,snrDb(s),method==4);
                if method==4
                    estimate=input.Truth;variance=zeros(size(estimate));
                else
                    network=[];blend=1;
                    if method==3,network=model.NetworkRX;blend=model.BlendRX*model.BlendMultiplier;end
                    [estimate,variance] = muCE(input.Samples,input.Noise,r.DownlinkLayout, ...
                        model.PriorsRX,r,names(method),network,blend,cfg.UERx);
                end
                [e,f] = muDecode(input,estimate,variance,payload,scene.Config,r);
                errors(packetIndex,s,precoder,estimator,:) = reshape(e,1,1,1,1,cfg.Users);
                failed(packetIndex,s,precoder,estimator,:) = reshape(f,1,1,1,1,cfg.Users);
                measured(s) = mean(input.Noise)/10^(-snrDb(s)/10);
            end
            calibration(packetIndex,precoder,estimator,:) = [d.MaximumPowerError ...
                d.MeanEstimatedLeakage d.MaximumCondition packet.PAPowerRatio mean(measured)];
        end
    end
    fprintf('Completed %d/%d MU packets, BS=%d, seed=%d\n',packetIndex,numPackets,cfg.Antennas,seed);
    clear scene packet input
end
result = struct('Config',cfg,'NumPackets',numPackets,'SNR',snrDb,'Schemes',schemes, ...
    'Estimators',names(methods),'EstimatorIndices',methods,'SlotErrors',errors,'CRCFailed',failed, ...
    'PayloadBits',r.PayloadBits,'CodedBits',r.CodedBits,'FrameDuration',r.FrameDuration, ...
    'Calibration',calibration,'SeedOffset',seedOffset,'BlendMultiplier',model.BlendMultiplier, ...
    'BlendTX',model.BlendTX,'BlendRX',model.BlendRX);
end
