function code1233_link(M, numSlots, outputFolder, arraySize, snrDb, testSeedOffset)
% Compare practical channel estimators and a perfect-reference receiver.
arguments
    M (1,1) double {mustBeMember(M,[256 512])} = 256
    numSlots (1,1) double {mustBeInteger,mustBePositive} = 32
    outputFolder (1,1) string = "results"
    arraySize (1,1) double {mustBeMember(arraySize,[32 64])} = 32
    snrDb (1,:) double {mustBeFinite} = [0:5, 10:5:40]
    testSeedOffset (1,1) double {mustBeInteger,mustBeNonnegative} = 0
end
assert(~isempty(snrDb) && all(diff(snrDb)>0),"SNR points must be increasing.");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

carrier = nrCarrierConfig('NSizeGrid',66,'SubcarrierSpacing',60,'CyclicPrefix','extended');
ofdm = nrOFDMInfo(carrier);
K = 12 * carrier.NSizeGrid;
L = carrier.SymbolsPerSlot;
slotSamples = sum(ofdm.SymbolLengths(1:L));
carrierFrequency = 28e9;
speedKmh = 120;
maxDoppler = (speedKmh / 3.6) * carrierFrequency / 299792458;
delaySpread = 1000e-9;
targetRate = 0.5;
trainingSNR = [0 5 15 30 40];
numTrainingSlots = 16;
bitsPerSymbol = log2(M);

% Reserve two symbols for control and transmit one-port PDSCH DM-RS.
pdsch = nrPDSCHConfig('PRBSet',0:carrier.NSizeGrid-1,'SymbolAllocation',[2 L-2], ...
    'MappingType','A','Modulation','256QAM','NumLayers',1);
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
pdsch.DMRS.DMRSAdditionalPosition = 3;
dataIndices = nrPDSCHIndices(carrier,pdsch);
pilotIndices = nrPDSCHDMRSIndices(carrier,pdsch);
pilotSymbols = nrPDSCHDMRS(carrier,pdsch);
assert(isempty(intersect(dataIndices,pilotIndices)));
pilotMask = false(K,L);
pilotMask(pilotIndices) = true;
prgWidth = 24; % Two resource blocks per precoding group.
pilot = pilotLayout(pilotMask,prgWidth);
[~,pilot.dataTimes] = ind2sub([K L],dataIndices);
[stencil,groups] = makeStencil(pilot);
codedBitsPerSlot = numel(dataIndices) * bitsPerSymbol;
payloadBitsPerSlot = 8 * floor(targetRate * codedBitsPerSlot / 8);
% MATLAB's NR LDPC rate matcher has no 512-QAM mode. Use its QPSK bit
% interleaver consistently at both ends for the experimental 512-QAM map.
rateMatchMod = '256QAM';
if M == 512
    rateMatchMod = 'QPSK';
end
encoder = nrDLSCH('TargetCodeRate',targetRate);
decoder = nrDLSCHDecoder('TargetCodeRate',targetRate, ...
    'TransportBlockLength',payloadBitsPerSlot,'MaximumLDPCIterationCount',20);

interleaverStream = RandStream('mt19937ar','Seed',1729);
bitOrder = randperm(interleaverStream,codedBitsPerSlot).';

% Check CRC, LDPC, interleaving, and QAM mapping as one noiseless chain.
testStream = RandStream('mt19937ar','Seed',2000);
testPayload = int8(randi(testStream,[0 1],payloadBitsPerSlot,1));
setTransportBlock(encoder,testPayload);
testCoded = encoder(rateMatchMod,1,codedBitsPerSlot,0);
testSymbols = qammod(testCoded(bitOrder),M,'InputType','bit','UnitAveragePower',true);
testHard = qamdemod(testSymbols,M,'OutputType','bit','UnitAveragePower',true);
assert(isequal(logical(testHard(:)),logical(testCoded(bitOrder))));
testLLR = zeros(codedBitsPerSlot,1);
testLLR(bitOrder) = 12*(1-2*double(testHard(:)));
[testDecoded,testCRCFailed] = decoder(testLLR,rateMatchMod,1,0);
assert(~testCRCFailed && all(testDecoded(:)==testPayload(:)),'Coded loopback failed.');
fprintf('Verified noiseless CRC, LDPC, interleaver and %d-QAM loopback.\n',M);

estimatorNames = {'LS','LMMSE','CNN','Perfect CSI + timing'};
slotErrors = zeros(numSlots,numel(snrDb),numel(estimatorNames));
txPowerPerDataRE = zeros(numSlots,1);
antennaSignalPowerPerDataRE = zeros(numSlots,1);
rfSignalPowerPerDataRE = zeros(numSlots,1);
rfNoisePowerAt0dB = zeros(numSlots,1);
c = 6+(arraySize==64);
Nt = arraySize;
Nr = arraySize;
numRF = 8;
phaseBits = 4;
soundingSNR = 15;
feedbackDelay = slotSamples/ofdm.SampleRate;
timingSearchSamples = ceil(12e-6*ofdm.SampleRate);
horizontalCorrelation = .45;
verticalCorrelation = .35;
arrayRows = arraySize/8;
rxFactor = arrayFactor(arrayRows,8,horizontalCorrelation,verticalCorrelation);
txFactor = rxFactor;
kSigned = (-K/2:K/2-1).';
phaseStep = 2*pi*kSigned/double(ofdm.Nfft);
timingAbsError = zeros(1,numel(snrDb));
configName = sprintf('%dx%d HBF',Nt,Nr);
fprintf('%d-QAM | %s | %d training slots and %d test slots\n',M,configName,numTrainingSlots,numSlots);
trainingPayload = RandStream('mt19937ar','Seed',60000+c);
trainingNoise = RandStream('mt19937ar','Seed',50000+c);
testPayloadStream = RandStream('mt19937ar','Seed',10000+c+testSeedOffset);
testNoise = RandStream('mt19937ar','Seed',20000+c+testSeedOffset);
trainingSounding = RandStream('mt19937ar','Seed',80000+c);
testSounding = RandStream('mt19937ar','Seed',90000+c+testSeedOffset);
trainingTruth = cell(numTrainingSlots,1);
trainingSamples = cell(numTrainingSlots,numel(trainingSNR));

probe = makeChannel(Nt,Nr,ofdm.SampleRate,delaySpread,maxDoppler);
probe.ChannelFiltering = false;
probe.OutputDataType = 'single';
probe.ChannelResponseOutput = 'ofdm-response';
channel = makeChannel(Nt,Nr,ofdm.SampleRate,delaySpread,maxDoppler);
channel.InitialTime = feedbackDelay;
channel.ChannelResponseOutput = 'ofdm-response';
channelInfo = info(channel);
tailSamples = max(channelInfo.MaximumChannelDelay,timingSearchSamples);
probe.NumTimeSamples = slotSamples + tailSamples;
for slot = 1:numTrainingSlots+numSlots
    training = slot <= numTrainingSlots;
    if training
        seed = 40000 + 1000*c + slot;
        payloadStream = trainingPayload;
        noiseStream = trainingNoise;
        soundingStream = trainingSounding;
    else
        seed = 30000 + 1000*c + testSeedOffset + slot-numTrainingSlots;
        payloadStream = testPayloadStream;
        noiseStream = testNoise;
        soundingStream = testSounding;
    end
    release(probe);
    release(channel);
    probe.Seed = seed;
    channel.Seed = seed;
    response = probe(carrier);
    assert(size(response,1)==K && size(response,3)==Nr && size(response,4)==Nt);
    Hsounding = correlateResponse(reshape(permute(response(:,1,:,:),[3 4 1 2]),Nr,Nt,K), ...
        rxFactor,txFactor);
    clear response
    soundingError = complex(randn(soundingStream,size(Hsounding),'single'), ...
        randn(soundingStream,size(Hsounding),'single'))/sqrt(2*10^(soundingSNR/10));
    [precoder,receiver] = beamformers(Hsounding+soundingError,numRF,numRF,prgWidth,phaseBits);
    clear Hsounding soundingError
    assert(max(abs(sum(abs(precoder).^2,1)-1)) < 2e-5);

    payload = int8(randi(payloadStream,[0 1],payloadBitsPerSlot,1));
    setTransportBlock(encoder,payload);
    coded = encoder(rateMatchMod,1,codedBitsPerSlot,0);
    layerGrid = complex(zeros(K,L));
    layerGrid(dataIndices) = qammod(coded(bitOrder),M,'InputType','bit','UnitAveragePower',true);
    layerGrid(pilotIndices) = pilotSymbols;
    antennaGrid = reshape(single(layerGrid),K,L,1).*reshape(precoder.',K,1,Nt);
    tx = nrOFDMModulate(carrier,antennaGrid,'Windowing',0);
    assert(size(tx,1)==slotSamples);
    [rx,response,timingOffset] = channel([tx*conj(txFactor); zeros(tailSamples,Nt,'like',tx)],carrier);
    if ~training
        trial = slot-numTrainingSlots;
        % The RF projection is orthonormal, so white noise has the same
        % variance before projection and on each retained RF output.
        assert(norm(double(receiver'*receiver)-eye(size(receiver,2)),'fro') < 1e-5);
        antennaWaveform = rx*rxFactor.';
    end
    rx = double(rx*(rxFactor.'*conj(receiver)));
    H = correlateResponse(permute(response(:,1:L,:,:),[3 4 1 2]),rxFactor,txFactor);
    clear response
    projected = pagemtimes(receiver','none',H,'none');
    Hsingle = pagemtimes(projected,reshape(precoder,Nt,1,K,1));
    Hgrid = double(reshape(permute(Hsingle,[3 4 1 2]),K,L,[]));
    clear H projected Hsingle

    baseNoise = (randn(noiseStream,size(rx)) + 1i*randn(noiseStream,size(rx))) ...
        /sqrt(2*double(ofdm.Nfft));
    oracleSamples = timingOffset+(1:slotSamples);
    if training
        cleanOffset = estimateTiming(carrier,rx,pilotIndices,pilotSymbols,timingSearchSamples);
        trainingTruth{slot} = single(Hgrid.*exp(1i*phaseStep*(cleanOffset-timingOffset)));
        for s = 1:numel(trainingSNR)
            noiseVariance = 10^(-trainingSNR(s)/10);
            noisyWaveform = rx+sqrt(noiseVariance)*baseNoise;
            offset = estimateTiming(carrier,noisyWaveform,pilotIndices,pilotSymbols,timingSearchSamples);
            receivedGrid = nrOFDMDemodulate(carrier,noisyWaveform(offset+(1:slotSamples),:));
            [ls,pilotGrid,nVar,scale] = practicalInputs(receivedGrid,pilotIndices,pilotSymbols,noiseVariance,pilot);
            timingPhase = single(exp(1i*phaseStep*(offset-cleanOffset)));
            trainingSamples{slot,s} = {single(ls),single(pilotGrid),nVar,scale,timingPhase};
        end
        if slot == numTrainingSlots
            lmmse = trainLMMSE(trainingTruth,stencil,groups,pilotIndices);
            cnn = trainCNN(trainingSamples,trainingTruth,lmmse,stencil,groups,pilot,c);
            clear trainingSamples trainingTruth
            fprintf('Training completed; starting independent evaluation.\n');
        end
        continue
    end

    txPowerPerDataRE(trial) = mean(sum(abs(antennaGrid(double(dataIndices) + (0:Nt-1)*K*L)).^2,2));
    antennaGridRx = nrOFDMDemodulate(carrier,antennaWaveform(oracleSamples,:));
    rfGrid = nrOFDMDemodulate(carrier,rx(oracleSamples,:));
    noiseGrid = nrOFDMDemodulate(carrier,baseNoise(oracleSamples,:));
    antennaSignalPowerPerDataRE(trial) = mean(abs(antennaGridRx(double(dataIndices) + (0:Nr-1)*K*L)).^2,'all');
    rfIndices = double(dataIndices) + (0:size(rx,2)-1)*K*L;
    rfSignalPowerPerDataRE(trial) = mean(abs(rfGrid(rfIndices)).^2,'all');
    rfNoisePowerAt0dB(trial) = mean(abs(noiseGrid(rfIndices)).^2,'all');
    assert(abs(10*log10(rfNoisePowerAt0dB(trial))) < 0.15,'Noise calibration failed.');
    for s = 1:numel(snrDb)
        noiseVariance = 10^(-snrDb(s)/10);
        noisyWaveform = rx+sqrt(noiseVariance)*baseNoise;
        offset = estimateTiming(carrier,noisyWaveform,pilotIndices,pilotSymbols,timingSearchSamples);
        timingAbsError(s) = timingAbsError(s)+abs(double(offset)-double(timingOffset));
        receivedGrid = nrOFDMDemodulate(carrier,noisyWaveform(offset+(1:slotSamples),:));
        oracleGrid = nrOFDMDemodulate(carrier,noisyWaveform(oracleSamples,:));
        [~,~,oracleNoise] = practicalInputs(oracleGrid,pilotIndices,pilotSymbols,noiseVariance,pilot);
        [ls,pilotGrid,nVar,scale] = practicalInputs(receivedGrid,pilotIndices,pilotSymbols,noiseVariance,pilot);
        wiener = estimateLMMSE(pilotGrid,nVar,scale,lmmse,stencil,groups);
        cnnEstimate = estimateCNN(cnn,ls,wiener,pilotGrid,scale,nVar,pilot);
        estimates = {ls,wiener,cnnEstimate,Hgrid};
        for est = 1:numel(estimatorNames)
            % The same equalizer, soft demapper and LDPC decoder are used.
            h = estimates{est};
            observed = receivedGrid;
            variance = nVar;
            if est == numel(estimatorNames)
                observed = oracleGrid;
                variance = oracleNoise;
            end
            gain = max(sum(abs(h).^2,3),eps);
            equalized = sum(conj(h).*observed,3)./gain;
            soft = qamdemod(equalized(dataIndices),M,'OutputType','approxllr', ...
                'UnitAveragePower',true,'NoiseVariance',1);
            pilotError = abs(equalized(pilotIndices)-pilotSymbols).^2;
            pilotGain = gain(pilotIndices);
            groupGain = max(pilot.timeGroups.'*pilotGain,eps);
            residualAtPilot = max(0,(pilot.timeGroups.'*(pilotGain.*pilotError) ...
                - variance*pilot.timeCounts)./groupGain);
            residual = pilot.timeInterp*residualAtPilot;
            dataVariance = variance./gain(dataIndices) + residual(pilot.dataTimes);
            weights = 1./reshape(dataVariance,1,[]);
            deinterleaved = zeros(codedBitsPerSlot,1);
            deinterleaved(bitOrder) = reshape(reshape(soft,bitsPerSymbol,[]).*weights,[],1);
            reset(decoder);
            decoded = decoder(max(-50,min(50,deinterleaved)),rateMatchMod,1,0);
            slotErrors(trial,s,est) = nnz(decoded(:) ~= payload(:));
        end
    end
    if mod(slot-numTrainingSlots,16) == 0 || slot == numTrainingSlots+numSlots
        fprintf('  %d/%d test slots\n',slot-numTrainingSlots,numSlots);
    end
end

postBitErrors = reshape(sum(slotErrors,1),numel(snrDb),numel(estimatorNames)).';
totalPayloadBits = repmat(payloadBitsPerSlot*numSlots,size(postBitErrors));
postBer = postBitErrors./totalPayloadBits;
timingMAE = timingAbsError/numSlots;
metrics = table(repelem(estimatorNames(:),numel(snrDb)),repmat(snrDb(:),numel(estimatorNames),1), ...
    reshape(postBer.',[],1),reshape(postBitErrors.',[],1),reshape(totalPayloadBits.',[],1), ...
    'VariableNames',{'Estimator','SNR_dB','PostLDPC_BER','PostLDPC_BitErrors','TotalPayloadBits'});
writetable(metrics,fullfile(outputFolder,'post_ldpc_ber.csv'));
[slotIndex,snrIndex,estimatorIndex] = ndgrid(1:numSlots,1:numel(snrDb),1:numel(estimatorNames));
estimatorColumn = estimatorNames(:);
snrColumn = snrDb(:);
slotMetrics = table(slotIndex(:),snrColumn(snrIndex(:)),estimatorColumn(estimatorIndex(:)), ...
    slotErrors(:),repmat(payloadBitsPerSlot,numel(slotErrors),1), ...
    'VariableNames',{'Slot','SNR_dB','Estimator','PostLDPC_BitErrors','PayloadBits'});
writetable(slotMetrics,fullfile(outputFolder,'slot_errors.csv'));
calibration = table((1:numSlots).',txPowerPerDataRE,antennaSignalPowerPerDataRE, ...
    rfSignalPowerPerDataRE,rfNoisePowerAt0dB,'VariableNames', ...
    {'Slot','TxPowerPerDataRE','AntennaSignalPowerPerDataRE','RFSignalPowerPerDataRE','RFNoisePowerAt0dB'});
writetable(calibration,fullfile(outputFolder,'snr_calibration.csv'));
save(fullfile(outputFolder,'post_ldpc_ber.mat'),'M','numSlots','testSeedOffset','numTrainingSlots','arraySize','numRF','prgWidth', ...
    'configName','estimatorNames','snrDb','postBer','postBitErrors','totalPayloadBits', ...
    'payloadBitsPerSlot','slotErrors','calibration','maxDoppler','delaySpread','speedKmh','carrierFrequency','targetRate','pilot', ...
    'phaseBits','soundingSNR','feedbackDelay','horizontalCorrelation','verticalCorrelation','timingMAE');
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
count = 4;
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
    model(g).R = (xx+xx')/(2*n);
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

function factor = arrayFactor(rows,columns,rhoX,rhoY)
vertical = toeplitz(rhoY.^(0:rows-1));
horizontal = toeplitz(rhoX.^(0:columns-1));
factor = single(chol(kron(horizontal,vertical),'lower'));
end

function h = correlateResponse(h,rxFactor,txFactor)
h = pagemtimes(rxFactor,pagemtimes(h,txFactor'));
end

function channel = makeChannel(Nt,Nr,Fs,delaySpread,maxDoppler)
channel = nrTDLChannel('DelayProfile','TDL-C','DelaySpread',delaySpread, ...
    'MaximumDopplerShift',maxDoppler,'SampleRate',Fs,'PathGainSampleRate','auto', ...
    'NumTransmitAntennas',Nt, ...
    'NumReceiveAntennas',Nr,'MIMOCorrelation','Low','NormalizePathGains',true, ...
    'NormalizeChannelOutputs',false,'RandomStream','mt19937ar with seed');
end

function [precoder,receiver] = beamformers(H,NrfTx,NrfRx,Wgroup,phaseBits)
[Nr,Nt,K] = size(H);
Rt = mean(pagemtimes(H,'ctranspose',H,'none'),3);
Rr = mean(pagemtimes(H,'none',H,'ctranspose'),3);
[Vt,Dt] = eig((Rt+Rt')/2,'vector');
[Vr,Dr] = eig((Rr+Rr')/2,'vector');
[~,txOrder] = sort(real(Dt),'descend');
[~,rxOrder] = sort(real(Dr),'descend');
step = 2*pi/2^phaseBits;
txPhase = step*round(angle(Vt(:,txOrder(1:NrfTx)))/step);
rxPhase = step*round(angle(Vr(:,rxOrder(1:NrfRx)))/step);
F = exp(1i*txPhase)/sqrt(Nt);
W = exp(1i*rxPhase)/sqrt(Nr);
receiver = orth(W);
effective = pagemtimes(receiver','none',H,'none');
effective = pagemtimes(effective,F);
precoder = complex(zeros(Nt,K,'like',H));
previous = [];
for first = 1:Wgroup:K
    block = first:min(first+Wgroup-1,K);
    groupH = effective(:,:,block);
    gram = mean(pagemtimes(groupH,'ctranspose',groupH,'none'),3);
    [V,D] = eig((gram+gram')/2,'vector');
    [~,best] = max(real(D));
    w = F*V(:,best);
    w = w/norm(w);
    if ~isempty(previous)
        overlap = previous'*w;
        if abs(overlap)>1e-6
            w = w*conj(overlap)/abs(overlap);
        end
    end
    precoder(:,block) = repmat(w,1,numel(block));
    previous = w;
end
end
