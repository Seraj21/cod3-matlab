function code1233_link(M,numSlots,outputFolder,arraySize,snrDb,testSeedOffset,options)
% Practical hybrid beamforming and matched post-LDPC channel-estimator comparison.
arguments
    M (1,1) double {mustBeMember(M,[256 512])} = 512
    numSlots (1,1) double {mustBeInteger,mustBePositive} = 32
    outputFolder (1,1) string = "results"
    arraySize (1,1) double {mustBeMember(arraySize,[32 64])} = 32
    snrDb (1,:) double {mustBeFinite} = [0:.5:10 11:20]
    testSeedOffset (1,1) double {mustBeInteger,mustBeNonnegative} = 0
    options.Mode (1,1) string {mustBeMember(options.Mode,["train" "evaluate"])} = "evaluate"
    options.ModelFile (1,1) string = ""
    options.NumRB (1,1) double {mustBeInteger,mustBePositive} = 66
    options.TrainingSlots (1,1) double {mustBeInteger,mustBePositive} = 24
    options.TestProfile (1,1) string = "CDL-C"
    options.RayFile (1,1) string = ""
    options.RayStart (1,1) double {mustBeInteger,mustBePositive} = 1
end
assert(~isempty(snrDb) && all(diff(snrDb)>0));
assert(snrDb(1)>=0 && snrDb(end)<=20,"The evaluation range is 0 to 20 dB.");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
carrier = nrCarrierConfig('NSizeGrid',options.NumRB,'SubcarrierSpacing',60,'CyclicPrefix','extended');
ofdm = nrOFDMInfo(carrier);
K = 12*carrier.NSizeGrid;
L = carrier.SymbolsPerSlot;
slotSamples = sum(ofdm.SymbolLengths(1:L));
pdsch = nrPDSCHConfig('PRBSet',0:carrier.NSizeGrid-1,'SymbolAllocation',[2 L-2], ...
    'MappingType','A','Modulation','256QAM','NumLayers',1,'EnablePTRS',true);
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
pdsch.DMRS.DMRSAdditionalPosition = 3;
pdsch.PTRS.TimeDensity = 1;
pdsch.PTRS.FrequencyDensity = 2;
dataIndices = nrPDSCHIndices(carrier,pdsch);
pilotIndices = nrPDSCHDMRSIndices(carrier,pdsch);
pilotSymbols = nrPDSCHDMRS(carrier,pdsch);
ptrsIndices = nrPDSCHPTRSIndices(carrier,pdsch);
ptrsSymbols = nrPDSCHPTRS(carrier,pdsch);
assert(isempty(intersect(dataIndices,pilotIndices)));
assert(isempty(intersect(dataIndices,ptrsIndices)));
mask = false(K,L);
mask(pilotIndices) = true;
pilot = pilotLayout(mask,24);
[~,pilot.dataTimes] = ind2sub([K L],dataIndices);
[stencil,groups] = makeStencil(pilot);
codedBitsPerSlot = numel(dataIndices)*log2(M);
payloadBitsPerSlot = 8*floor(.5*codedBitsPerSlot/8);
codec = ExperimentalLDPC(payloadBitsPerSlot,codedBitsPerSlot,log2(M),.5);
orderStream = RandStream('mt19937ar','Seed',1729);
bitOrder = randperm(orderStream,codedBitsPerSlot).';
c = 6+(arraySize==64);
settings = struct('ArraySize',arraySize,'CarrierFrequency',28e9,'PhaseBits',4, ...
    'PRGWidth',24,'SoundingSNR',15,'SampleRate',ofdm.SampleRate, ...
    'PhaseNoiseLevel',[-85 -100 -115],'PhaseNoiseOffset',[1e4 1e5 1e6], ...
    'InputBackoff',10,'AMPMDegrees',3,'FrequencyOffset',2800);
config = struct('Version',2,'M',M,'ArraySize',arraySize,'NumRB',options.NumRB, ...
    'Settings',settings,'PayloadBits',payloadBitsPerSlot,'CodedBits',codedBitsPerSlot);
modelFile = options.ModelFile;
if modelFile == ""
    modelFile = fullfile(outputFolder,sprintf('receiver_model_%d.mat',arraySize));
end
trainingSlots = options.TrainingSlots;
if options.Mode == "evaluate" && isfile(modelFile)
    model = load(modelFile,'cnn','lmmse','config','trainingProfiles','trainingSNR');
    assert(isequaln(model.config,config),"Cached model configuration does not match.");
    cnn = model.cnn;
    lmmse = model.lmmse;
    trainingProfiles = model.trainingProfiles;
    trainingSNR = model.trainingSNR;
    trainingSlots = 0;
else
    trainingProfiles = ["TDL-A" "TDL-B" "CDL-A" "CDL-B"];
    trainingSNR = [0 5 10 15 20];
end
testSlots = numSlots;
if options.Mode == "train"
    testSlots = 0;
end
trainingTruth = cell(trainingSlots,1);
trainingSamples = cell(trainingSlots,numel(trainingSNR));
choiceStream = RandStream('mt19937ar','Seed',110000+c);
trainingPayload = RandStream('mt19937ar','Seed',60000+c);
trainingNoise = RandStream('mt19937ar','Seed',50000+c);
payloadStream = RandStream('mt19937ar','Seed',10000+c+testSeedOffset);
testNoise = RandStream('mt19937ar','Seed',20000+c+testSeedOffset);
names = {'LS','LMMSE','CNN','Perfect CSI + timing'};
slotErrors = zeros(testSlots,numel(snrDb),4);
crcErrors = false(testSlots,numel(snrDb),4);
calibration = zeros(testSlots,7);
timingAbsError = zeros(testSlots,numel(snrDb));
cfoResidual = zeros(testSlots,numel(snrDb));
kSigned = (-K/2:K/2-1).';
phaseStep = 2*pi*kSigned/double(ofdm.Nfft);
fprintf('Practical HBF %dx%d | %d-QAM | %d training, %d test slots\n', ...
    arraySize,arraySize,M,trainingSlots,testSlots);
for slot = 1:trainingSlots+testSlots
    training = slot<=trainingSlots;
    if training
        profile = trainingProfiles(mod(slot-1,numel(trainingProfiles))+1);
        spread = [100 300 600 1000]*1e-9;
        spread = spread(randi(choiceStream,4));
        speed = 120*rand(choiceStream);
        seed = 40000+1000*c+slot;
        payload = int8(randi(trainingPayload,[0 1],payloadBitsPerSlot,1));
        noiseStream = trainingNoise;
    else
        trial = slot-trainingSlots;
        profile = options.TestProfile;
        spread = 1e-6;
        speed = 120;
        seed = 30000+1000*c+testSeedOffset+trial;
        payload = int8(randi(payloadStream,[0 1],payloadBitsPerSlot,1));
        noiseStream = testNoise;
    end
    coded = codec.encode(payload);
    layerGrid = complex(zeros(K,L));
    layerGrid(dataIndices) = qammod(coded(bitOrder),M,'InputType','bit','UnitAveragePower',true);
    layerGrid(pilotIndices) = pilotSymbols;
    layerGrid(ptrsIndices) = ptrsSymbols;
    packetSettings = settings;
    if profile=="DeepMIMO"
        packetSettings.RayFile = options.RayFile;
        packetSettings.RayIndex = options.RayStart+trial-1;
    end
    packet = practicalPacket(carrier,layerGrid,packetSettings,profile,spread,speed,seed);
    rx = packet.Waveform;
    baseNoise = complex(randn(noiseStream,size(rx)),randn(noiseStream,size(rx))) ...
        /sqrt(2*double(ofdm.Nfft));
    points = snrDb;
    if training
        points = trainingSNR;
        cleanOffset = estimateTiming(carrier,rx,pilotIndices,pilotSymbols,ceil(12e-6*ofdm.SampleRate));
        trainingTruth{slot} = single(packet.Channel.*exp(1i*phaseStep*(cleanOffset-packet.TimingOffset)));
    else
        white = nrOFDMDemodulate(carrier,baseNoise(packet.TimingOffset+(1:slotSamples),:));
        clean = nrOFDMDemodulate(carrier,rx(packet.TimingOffset+(1:slotSamples),:));
        noisePower = mean(abs(white).^2,'all');
        assert(abs(10*log10(noisePower))<.15);
        calibration(trial,:) = [trial packet.TxPower packet.PAOutputPower ...
            packet.AntennaPower mean(abs(clean).^2,'all') noisePower packet.SelectedBank];
    end
    for s = 1:numel(points)
        variance = 10^(-points(s)/10);
        noisy = rx+sqrt(variance)*baseNoise;
        offset = estimateTiming(carrier,noisy,pilotIndices,pilotSymbols,ceil(12e-6*ofdm.SampleRate));
        frequency = estimateCFO(noisy,offset,ofdm,L);
        compensated = noisy.*exp(-1i*2*pi*frequency*(0:size(noisy,1)-1).'/ofdm.SampleRate);
        received = nrOFDMDemodulate(carrier,compensated(offset+(1:slotSamples),:));
        [ls,pilotGrid,nVar,scale] = practicalInputs(received,pilotIndices,pilotSymbols,variance,pilot);
        residualPhase = exp(-1i*2*pi*frequency*packet.SymbolCenters/ofdm.SampleRate);
        if training
            % Targets retain RF phase. CFO compensation is known to this receiver.
            correction = exp(1i*phaseStep*(offset-cleanOffset)).*residualPhase;
            trainingSamples{slot,s} = {single(ls),single(pilotGrid),nVar,scale,single(correction)};
            continue
        end
        timingAbsError(trial,s) = abs(offset-packet.TimingOffset);
        cfoResidual(trial,s) = settings.FrequencyOffset-frequency;
        wiener = estimateLMMSE(pilotGrid,nVar,scale,lmmse,stencil,groups);
        neural = estimateCNN(cnn,ls,wiener,pilotGrid,scale,nVar,pilot);
        oracle = packet.Channel.*reshape(residualPhase,1,L,1);
        estimates = {ls,wiener,neural,oracle};
        for method = 1:4
            observed = received;
            methodNoise = nVar;
            if method==4
                observed = nrOFDMDemodulate(carrier,compensated( ...
                    packet.TimingOffset+(1:slotSamples),:));
                [~,~,methodNoise] = practicalInputs(observed,pilotIndices,pilotSymbols,variance,pilot);
            end
            h = estimates{method};
            gain = max(sum(abs(h).^2,3),eps);
            equalized = sum(conj(h).*observed,3)./gain;
            equalized = correctPTRS(equalized,ptrsIndices,ptrsSymbols);
            soft = qamdemod(equalized(dataIndices),M,'OutputType','approxllr', ...
                'UnitAveragePower',true,'NoiseVariance',1);
            pilotError = abs(equalized(pilotIndices)-pilotSymbols).^2;
            pilotGain = gain(pilotIndices);
            groupGain = max(pilot.timeGroups.'*pilotGain,eps);
            residualAtPilot = max(0,(pilot.timeGroups.'*(pilotGain.*pilotError) ...
                -methodNoise*pilot.timeCounts)./groupGain);
            residual = pilot.timeInterp*residualAtPilot;
            dataVariance = methodNoise./gain(dataIndices)+residual(pilot.dataTimes);
            weighted = reshape(reshape(soft,log2(M),[])./reshape(dataVariance,1,[]),[],1);
            deinterleaved = zeros(codedBitsPerSlot,1);
            deinterleaved(bitOrder) = weighted;
            [decoded,failed] = codec.decode(max(-50,min(50,deinterleaved)));
            slotErrors(trial,s,method) = nnz(decoded(:)~=payload(:));
            crcErrors(trial,s,method) = failed;
        end
    end
    if training && slot==trainingSlots
        lmmse = trainLMMSE(trainingTruth,stencil,groups,pilotIndices);
        cnn = trainCNN(trainingSamples,trainingTruth,lmmse,stencil,groups,pilot,c);
        save(modelFile,'cnn','lmmse','config','trainingProfiles','trainingSNR','-v7.3');
        clear trainingSamples trainingTruth
        fprintf('Frozen model saved. Held-out evaluation starts.\n');
    end
    if ~training && (mod(trial,8)==0 || trial==testSlots)
        fprintf('Completed %d/%d test slots\n',trial,testSlots);
    end
end
if options.Mode=="train"
    return
end
cnnFingerprint = cellfun(@(w)gather(extractdata(w)),cnn.Learnables.Value,'UniformOutput',false);
postBitErrors = reshape(sum(slotErrors,1),numel(snrDb),4).';
totalPayloadBits = repmat(payloadBitsPerSlot*testSlots,size(postBitErrors));
postBer = postBitErrors./totalPayloadBits;
[trials,snrIndex,methodIndex] = ndgrid(1:testSlots,1:numel(snrDb),1:4);
nameColumn = names(:);
snrColumn = snrDb(:);
slots = table(trials(:),snrColumn(snrIndex(:)),nameColumn(methodIndex(:)), ...
    slotErrors(:),repmat(payloadBitsPerSlot,numel(slotErrors),1), ...
    'VariableNames',{'Slot','SNR_dB','Estimator','PostLDPC_BitErrors','PayloadBits'});
writetable(slots,fullfile(outputFolder,'slot_errors.csv'));
metrics = table(repelem(names(:),numel(snrDb)),repmat(snrDb(:),4,1), ...
    reshape(postBer.',[],1),reshape(postBitErrors.',[],1),reshape(totalPayloadBits.',[],1), ...
    'VariableNames',{'Estimator','SNR_dB','PostLDPC_BER','PostLDPC_BitErrors','TotalPayloadBits'});
writetable(metrics,fullfile(outputFolder,'post_ldpc_ber.csv'));
calibration = array2table(calibration,'VariableNames',{'Slot','TxMeanGridPower', ...
    'PAOutputPowerRatio','AntennaSignalPowerPerRE','RFSignalPowerPerRE','RFNoisePowerAt0dB','SelectedBank'});
writetable(calibration,fullfile(outputFolder,'snr_calibration.csv'));
save(fullfile(outputFolder,'post_ldpc_ber.mat'),'M','arraySize','numSlots','snrDb', ...
    'names','slotErrors','postBitErrors','postBer','totalPayloadBits','payloadBitsPerSlot', ...
    'cnnFingerprint','settings','config','trainingProfiles','calibration','crcErrors', ...
    'timingAbsError','cfoResidual','testSeedOffset');
end

function equalized = correctPTRS(equalized,indices,symbols)
[K,L] = size(equalized);
[~,times] = ind2sub([K L],indices);
phasor = accumarray(times,equalized(indices).*conj(symbols),[L 1]);
valid = find(abs(phasor)>0);
if numel(valid)>1
    phase = interp1(valid,unwrap(angle(phasor(valid))), ...
        min(valid(end),max(valid(1),(1:L).')),'linear');
    equalized = equalized.*exp(-1i*phase.');
end
end

function p = pilotLayout(mask,W)
[K,L] = size(mask);
assert(mod(K,W)==0);
G = K/W;
[f,t] = find(mask);
times = unique(t);
assert(numel(times)>=3,'Expected at least three time-domain DM-RS positions.');
frequencies = find(mask(1:W,times(1)));
assert(isequal(mask,repmat(mask(1:W,:),G,1)));
assert(all(mask(:,times)==mask(:,times(1)),'all'));
p.W = W;
p.K = K;
p.L = L;
p.G = G;
p.P = numel(frequencies);
p.T = numel(times);
p.freq = frequencies;
p.times = times;
p.timeGroups = double(t==times.');
p.timeCounts = sum(p.timeGroups,1).';
p.timeInterp = interp1(times,eye(p.T),min(times(end),max(times(1),(1:L).')),'linear');
p.freqInterp = interp1(frequencies,eye(p.P), ...
    min(frequencies(end),max(frequencies(1),(1:W).')),'linear');
p.mask = mask(1:W,:);
assert(numel(f)==p.P*G*p.T);
fprintf('DM-RS OFDM symbols (zero based): %s\n',mat2str(times.'-1));
end

function [ls,pilotGrid,nVar,scale] = practicalInputs(grid,pilotIndices,pilotSymbols,nominalNoise,p)
R = size(grid,3);
flat = reshape(grid,p.K*p.L,R);
pilotLS = flat(pilotIndices,:)./pilotSymbols;
pilotGrid = complex(zeros(p.K*p.L,R));
pilotGrid(pilotIndices,:) = pilotLS;
samples = reshape(pilotLS,p.P,p.G,p.T,R);
% Leave the middle pilot out. The 1.5 factor is the variance gain of
% subtracting the average of two independent noisy pilot observations.
predictionError = samples(2:end-1,:,:,:)-(samples(1:end-2,:,:,:)+samples(3:end,:,:,:))/2;
pilotDisturbance = median(abs(predictionError).^2,'all')/(1.5*log(2));
nVar = max(nominalNoise,pilotDisturbance);
scale = sqrt(max(mean(abs(pilotLS).^2,1)-nVar,max(0.1*nVar,1e-4)));
inPRGs = permute(samples,[1 3 2 4]);
freq = pagemtimes(p.freqInterp,inPRGs);
both = pagemtimes(freq,p.timeInterp.');
ls = reshape(permute(both,[1 3 2 4]),p.K,p.L,R);
end

function [indices,groups] = makeStencil(p)
k = (1:p.K).';
localK = mod(k-1,p.W)+1;
base = k-localK;
count = 6;
center = interp1(double(p.freq),1:p.P,localK,'nearest','extrap');
start = max(1,min(p.P-count+1,center-1));
neighbors = p.freq(start+(0:count-1));
indices = zeros(p.K,p.T*count);
for t = 1:p.T
    indices(:,(t-1)*count+(1:count)) = base+neighbors+(p.times(t)-1)*p.K;
end
[~,~,groups] = unique(neighbors-localK,'rows');
end

function model = trainLMMSE(truth,stencil,groups,pilotIndices)
[K,L,~] = size(truth{1});
width = size(stencil,2);
for g = 1:max(groups)
    rows = find(groups==g);
    xx = zeros(width);
    xy = zeros(width,L);
    n = 0;
    for slot = 1:numel(truth)
        h = double(truth{slot});
        for branch = 1:size(h,3)
            H = h(:,:,branch);
            H = H / sqrt(max(mean(abs(H(pilotIndices)).^2),1e-4));
            x = reshape(H(stencil(rows,:)),numel(rows),width);
            y = H(rows,:);
            xx = xx+x'*x;
            xy = xy+x'*y;
            n = n+numel(rows);
        end
    end
    covariance = (xx+xx')/(2*n);
    shrinkage = min(.05,width/n);
    model(g).R = (1-shrinkage)*covariance+shrinkage*real(trace(covariance))/width*eye(width);
    model(g).C = xy/n;
end
assert(size(stencil,1)==K);
end

function h = estimateLMMSE(pilots,nVar,scale,model,stencil,groups)
K = size(stencil,1);
L = size(model(1).C,2);
R = size(pilots,2);
width = size(stencil,2);
h = complex(zeros(K,L,R));
for branch = 1:R
    values = pilots(:,branch);
    regularization = (nVar/scale(branch)^2+1e-3)*eye(width);
    for g = 1:numel(model)
        rows = find(groups==g);
        x = reshape(values(stencil(rows,:)),numel(rows),width);
        h(rows,:,branch) = x*((model(g).R+regularization)\model(g).C);
    end
end
end

function x = makeFeatures(ls,wiener,pilots,scale,nVar,p)
R = size(ls,3);
divisor = reshape(scale,1,1,R);
ls = asPRGPatches(ls./divisor,p);
wiener = asPRGPatches(wiener./divisor,p);
sparse = asPRGPatches(reshape(pilots,p.K,p.L,R)./divisor,p);
relativeNoise = repelem(log10(max(1e-5,nVar./scale.^2)),p.G);
timeCoordinate = repmat(linspace(-1,1,p.L),p.W,1,1,p.G*R);
x = single(cat(3,real(ls),imag(ls),real(wiener),imag(wiener),real(sparse),imag(sparse), ...
    repmat(p.mask,1,1,1,p.G*R), ...
    repmat(reshape(relativeNoise,1,1,1,[]),p.W,p.L,1,1),timeCoordinate));
end

function patches = asPRGPatches(h,p)
patches = reshape(permute(reshape(h,p.W,p.G,p.L,[]),[1 3 2 4]),p.W,p.L,1,[]);
end

function net = trainCNN(samples,truth,lmmse,stencil,groups,p,c)
X = cell(size(samples));
Y = cell(size(samples));
for slot = 1:size(samples,1)
    % Sample across frequency and across RF branches, with fixed selection.
    bands = mod((slot-1)+[0 5 10 15 20 25],p.G)+1;
    R = size(truth{slot},3);
    branches = unique(mod((slot-1)+[0 3 6],R)+1);
    selected = reshape(bands.'+(branches-1)*p.G,1,[]);
    for s = 1:size(samples,2)
        item = samples{slot,s};
        ls = double(item{1});
        pilots = double(item{2});
        nVar = item{3};
        scale = item{4};
        wiener = estimateLMMSE(pilots,nVar,scale,lmmse,stencil,groups);
        features = makeFeatures(ls,wiener,pilots,scale,nVar,p);
        target = (double(truth{slot}).*double(item{5})-wiener)./reshape(scale,1,1,[]);
        target = asPRGPatches(target,p);
        X{slot,s} = features(:,:,:,selected);
        Y{slot,s} = single(cat(3,real(target(:,:,:,selected)),imag(target(:,:,:,selected))));
    end
end
rng(70000+c,'twister');
layers = [
    imageInputLayer([p.W p.L 9],'Normalization','none')
    convolution2dLayer([5 3],16,'Padding','same')
    reluLayer
    convolution2dLayer([5 3],16,'Padding','same','DilationFactor',[1 2])
    reluLayer
    convolution2dLayer([3 3],2,'Padding','same','WeightsInitializer','zeros','BiasInitializer','zeros')
];
opts = trainingOptions('adam','MaxEpochs',10,'MiniBatchSize',48,'InitialLearnRate',8e-4, ...
    'Shuffle','every-epoch','ExecutionEnvironment','cpu','Verbose',false);
net = trainnet(cat(4,X{:}),cat(4,Y{:}),layers,'mse',opts);
end

function h = estimateCNN(net,ls,wiener,pilots,scale,nVar,p)
R = size(ls,3);
x = makeFeatures(ls,wiener,pilots,scale,nVar,p);
z = double(extractdata(predict(net,dlarray(x,'SSCB'))));
correction = reshape(complex(z(:,:,1,:),z(:,:,2,:)),p.W,p.L,p.G,R);
correction = reshape(permute(correction,[1 3 2 4]),p.K,p.L,R);
h = wiener+correction.*reshape(scale,1,1,R);
end

function offset = estimateTiming(carrier,waveform,indices,symbols,window)
[~,mag] = nrTimingEstimate(carrier,waveform,indices,symbols);
energy = sum(abs(mag(1:min(window+1,size(mag,1)),:)).^2,2);
[~,peak] = max(energy);
offset = peak-1;
end

