function code1233_link(M, numSlots, outputFolder, configIndex)
% Post-LDPC BER for a high-mobility, long-delay TDL-C channel with perfect CSI.
arguments
    M (1,1) double {mustBeMember(M,[256 512])} = 256
    numSlots (1,1) double {mustBeInteger,mustBePositive} = 64
    outputFolder (1,1) string = "results"
    configIndex (1,1) double {mustBeInteger,mustBeInRange(configIndex,1,6)} = 6
end
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
snrDb = [0:5, 10:5:40];
trainingSNR = [0 5 15 30 40];
numTrainingSlots = 16;
bitsPerSymbol = log2(M);

% Reserve two symbols for control and transmit one-port PDSCH DM-RS.
pdsch = nrPDSCHConfig('PRBSet',0:carrier.NSizeGrid-1,'SymbolAllocation',[2 L-2], ...
    'MappingType','A','Modulation','256QAM','NumLayers',1);
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
pdsch.DMRS.DMRSAdditionalPosition = 1;
dataIndices = nrPDSCHIndices(carrier,pdsch);
pilotIndices = nrPDSCHDMRSIndices(carrier,pdsch);
pilotSymbols = nrPDSCHDMRS(carrier,pdsch);
assert(isempty(intersect(dataIndices,pilotIndices)));
pilotMask = false(K,L);
pilotMask(pilotIndices) = true;
[stencil,groups] = makeStencil(pilotMask);
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

configs = [1 1 1 1; 1 4 1 4; 4 1 4 1; 4 4 4 4; 16 16 4 4; 32 32 8 8];
configNames = {'SISO (1x1)','SIMO (1x4)','MISO (4x1)','MIMO (4x4)', ...
    'mMIMO (16x16 HBF)','mMIMO (32x32 HBF)'};
estimatorNames = {'LS','LMMSE','CNN','Perfect CSI'};
postBitErrors = zeros(numel(estimatorNames),numel(snrDb));
c = configIndex;
Nt = configs(c,1);
Nr = configs(c,2);
fprintf('%d-QAM | %s | %d training slots and %d test slots\n',M,configNames{c},numTrainingSlots,numSlots);
trainingPayload = RandStream('mt19937ar','Seed',60000+c);
trainingNoise = RandStream('mt19937ar','Seed',50000+c);
testPayloadStream = RandStream('mt19937ar','Seed',10000+c);
testNoise = RandStream('mt19937ar','Seed',20000+c);
trainingTruth = cell(numTrainingSlots,1);
trainingInputs = cell(numTrainingSlots,numel(trainingSNR));
trainingTargets = cell(size(trainingInputs));

probe = makeChannel(Nt,Nr,ofdm.SampleRate,delaySpread,maxDoppler);
probe.ChannelFiltering = false;
probe.ChannelResponseOutput = 'ofdm-response';
channel = makeChannel(Nt,Nr,ofdm.SampleRate,delaySpread,maxDoppler);
channelInfo = info(channel);
tailSamples = channelInfo.MaximumChannelDelay;
probe.NumTimeSamples = slotSamples + tailSamples;
for slot = 1:numTrainingSlots+numSlots
    training = slot <= numTrainingSlots;
    if training
        seed = 40000 + 1000*c + slot;
        payloadStream = trainingPayload;
        noiseStream = trainingNoise;
    else
        seed = 30000 + 1000*c + slot-numTrainingSlots;
        payloadStream = testPayloadStream;
        noiseStream = testNoise;
    end
    release(probe);
    release(channel);
    probe.Seed = seed;
    channel.Seed = seed;
    [response,timingOffset] = probe(carrier);
    assert(size(response,1) == K && size(response,2) >= L && ...
        size(response,3) == Nr && size(response,4) == Nt);
    H = permute(response(:,1:L,:,:),[3 4 1 2]);
    [precoder,receiver] = beamformers(H(:,:,:,1),configs(c,3),configs(c,4));

    payload = int8(randi(payloadStream,[0 1],payloadBitsPerSlot,1));
    setTransportBlock(encoder,payload);
    coded = encoder(rateMatchMod,1,codedBitsPerSlot,0);
    layerGrid = complex(zeros(K,L));
    layerGrid(dataIndices) = qammod(coded(bitOrder),M,'InputType','bit','UnitAveragePower',true);
    layerGrid(pilotIndices) = pilotSymbols;
    antennaGrid = reshape(layerGrid,K,L,1).*reshape(precoder.',K,1,Nt);
    tx = nrOFDMModulate(carrier,antennaGrid,'Windowing',0);
    assert(size(tx,1)==slotSamples);
    rx = channel([tx; zeros(tailSamples,Nt)]);
    rx = rx(timingOffset+(1:slotSamples),:)*conj(receiver);
    projected = pagemtimes(receiver','none',H,'none');
    Hsingle = pagemtimes(projected,reshape(precoder,Nt,1,K,1));
    Hgrid = reshape(permute(Hsingle,[3 4 1 2]),K,L,[]);

    baseNoise = (randn(noiseStream,size(rx)) + 1i*randn(noiseStream,size(rx))) ...
        /sqrt(2*double(ofdm.Nfft));
    signalGrid = nrOFDMDemodulate(carrier,rx);
    noiseGrid = nrOFDMDemodulate(carrier,baseNoise);
    if training
        % Select two RF branches per slot for CNN training; LMMSE uses all.
        trainBranches = 1:min(2,size(Hgrid,3));
        trainingTruth{slot} = single(Hgrid);
        for s = 1:numel(trainingSNR)
            noiseVariance = 10^(-trainingSNR(s)/10);
            receivedGrid = signalGrid + sqrt(noiseVariance)*noiseGrid;
            [ls,~,nVar,scale] = practicalInputs(carrier,receivedGrid,pilotIndices,pilotSymbols,noiseVariance);
            x = makeFeatures(ls(:,:,trainBranches),scale(trainBranches),nVar,pilotMask);
            truth = Hgrid(:,:,trainBranches)./reshape(scale(trainBranches),1,1,[]);
            residual = truth-ls(:,:,trainBranches)./reshape(scale(trainBranches),1,1,[]);
            trainingInputs{slot,s} = x;
            trainingTargets{slot,s} = single(cat(3, ...
                reshape(real(residual),K,L,1,[]),reshape(imag(residual),K,L,1,[])));
        end
        if slot == numTrainingSlots
            lmmse = trainLMMSE(trainingTruth,stencil,groups,pilotIndices);
            cnn = trainCNN(cat(4,trainingInputs{:}),cat(4,trainingTargets{:}),K,L,c);
            clear trainingInputs trainingTargets trainingTruth
            fprintf('Training completed; starting independent evaluation.\n');
        end
        continue
    end

    for s = 1:numel(snrDb)
        noiseVariance = 10^(-snrDb(s)/10);
        receivedGrid = signalGrid + sqrt(noiseVariance)*noiseGrid;
        [ls,pilotGrid,nVar,scale] = practicalInputs(carrier,receivedGrid,pilotIndices,pilotSymbols,noiseVariance);
        wiener = estimateLMMSE(pilotGrid,nVar,scale,lmmse,stencil,groups);
        cnnEstimate = estimateCNN(cnn,ls,scale,nVar,pilotMask);
        estimates = {ls,wiener,cnnEstimate,Hgrid};
        for est = 1:numel(estimatorNames)
            % The same equalizer, soft demapper and LDPC decoder are used.
            h = estimates{est};
            gain = max(sum(abs(h).^2,3),eps);
            equalized = sum(conj(h).*receivedGrid,3)./gain;
            soft = qamdemod(equalized(dataIndices),M,'OutputType','approxllr', ...
                'UnitAveragePower',true,'NoiseVariance',1);
            weights = reshape(gain(dataIndices),1,[])/nVar;
            deinterleaved = zeros(codedBitsPerSlot,1);
            deinterleaved(bitOrder) = reshape(reshape(soft,bitsPerSymbol,[]).*weights,[],1);
            reset(decoder);
            decoded = decoder(max(-50,min(50,deinterleaved)),rateMatchMod,1,0);
            postBitErrors(est,s) = postBitErrors(est,s)+nnz(decoded(:) ~= payload(:));
        end
    end
    if mod(slot-numTrainingSlots,16) == 0 || slot == numTrainingSlots+numSlots
        fprintf('  %d/%d test slots\n',slot-numTrainingSlots,numSlots);
    end
end

totalPayloadBits = repmat(payloadBitsPerSlot*numSlots,size(postBitErrors));
postBer = postBitErrors./totalPayloadBits;
metrics = table(repelem(estimatorNames(:),numel(snrDb)),repmat(snrDb(:),numel(estimatorNames),1), ...
    reshape(postBer.',[],1),reshape(postBitErrors.',[],1),reshape(totalPayloadBits.',[],1), ...
    'VariableNames',{'Estimator','SNR_dB','PostLDPC_BER','PostLDPC_BitErrors','TotalPayloadBits'});
writetable(metrics,fullfile(outputFolder,'post_ldpc_ber.csv'));
configName = configNames{c};
save(fullfile(outputFolder,'post_ldpc_ber.mat'),'M','numSlots','numTrainingSlots','configIndex', ...
    'configName','estimatorNames','snrDb','postBer','postBitErrors','totalPayloadBits', ...
    'payloadBitsPerSlot','maxDoppler','delaySpread','speedKmh','carrierFrequency','targetRate');
end

function [ls,pilotGrid,nVar,scale] = practicalInputs(carrier,grid,pilotIndices,pilotSymbols,nominalNoise)
[K,L,R] = size(grid);
[ls,estimatedNoise] = nrChannelEstimate(carrier,grid,pilotIndices,pilotSymbols);
ls = reshape(ls,K,L,R);
flat = reshape(grid,K*L,R);
pilotLS = flat(pilotIndices,:)./pilotSymbols;
pilotGrid = complex(zeros(K*L,R));
pilotGrid(pilotIndices,:) = pilotLS;
nVar = max(nominalNoise,mean(double(estimatedNoise(:))));
scale = sqrt(max(mean(abs(pilotLS).^2,1)-nVar,max(0.1*nVar,1e-4)));
end

function [indices,groups] = makeStencil(mask)
[K,~] = size(mask);
times = find(any(mask,1));
pilots = find(mask(:,times(1)));
assert(all(mask(:,times)==mask(:,times(1)),'all'));
count = 4;
center = interp1(double(pilots),1:numel(pilots), (1:K).','nearest','extrap');
start = max(1,min(numel(pilots)-count+1,center-1));
neighbors = pilots(start+(0:count-1));
indices = zeros(K,numel(times)*count);
for t = 1:numel(times)
    indices(:,(t-1)*count+(1:count)) = neighbors+(times(t)-1)*K;
end
[~,~,groups] = unique(neighbors-(1:K).','rows');
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

function x = makeFeatures(ls,scale,nVar,pilotMask)
[K,L,R] = size(ls);
shape = [K L 1 R];
scaled = ls./reshape(scale,1,1,R);
relativeNoise = log10(max(1e-5,nVar./scale.^2));
timeCoordinate = repmat(linspace(-1,1,L),K,1,1,R);
x = single(cat(3,reshape(real(scaled),shape),reshape(imag(scaled),shape), ...
    repmat(pilotMask,1,1,1,R),repmat(reshape(relativeNoise,1,1,1,R),K,L,1,1),timeCoordinate));
end

function net = trainCNN(x,y,K,L,c)
rng(70000+c,'twister');
layers = [
    imageInputLayer([K L 5],'Normalization','none')
    convolution2dLayer([5 3],12,'Padding','same')
    reluLayer
    convolution2dLayer([5 3],12,'Padding','same','DilationFactor',[1 2])
    reluLayer
    convolution2dLayer([3 3],2,'Padding','same')
];
opts = trainingOptions('adam','MaxEpochs',6,'MiniBatchSize',8,'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch','ExecutionEnvironment','cpu','Verbose',false);
net = trainnet(x,y,layers,'mse',opts);
end

function h = estimateCNN(net,ls,scale,nVar,pilotMask)
[K,L,R] = size(ls);
x = makeFeatures(ls,scale,nVar,pilotMask);
correction = double(extractdata(predict(net,dlarray(x,'SSCB'))));
residual = reshape(complex(correction(:,:,1,:),correction(:,:,2,:)),K,L,R);
h = ls+residual.*reshape(scale,1,1,R);
end

function channel = makeChannel(Nt,Nr,Fs,delaySpread,maxDoppler)
channel = nrTDLChannel('DelayProfile','TDL-C','DelaySpread',delaySpread, ...
    'MaximumDopplerShift',maxDoppler,'SampleRate',Fs,'PathGainSampleRate','auto', ...
    'NumTransmitAntennas',Nt, ...
    'NumReceiveAntennas',Nr,'MIMOCorrelation','Low','NormalizePathGains',true, ...
    'NormalizeChannelOutputs',false,'RandomStream','mt19937ar with seed');
end

function [precoder,receiver] = beamformers(H,NrfTx,NrfRx)
[Nr,Nt,K] = size(H);
Rt = mean(pagemtimes(H,'ctranspose',H,'none'),3);
Rr = mean(pagemtimes(H,'none',H,'ctranspose'),3);
[Vt,Dt] = eig((Rt+Rt')/2,'vector');
[Vr,Dr] = eig((Rr+Rr')/2,'vector');
[~,txOrder] = sort(real(Dt),'descend');
[~,rxOrder] = sort(real(Dr),'descend');
F = exp(1i*angle(Vt(:,txOrder(1:NrfTx))))/sqrt(Nt);
W = exp(1i*angle(Vr(:,rxOrder(1:NrfRx))))/sqrt(Nr);
receiver = orth(W);
effective = pagemtimes(receiver','none',H,'none');
effective = pagemtimes(effective,F);
precoder = complex(zeros(Nt,K));
for k = 1:K
    [~,~,V] = svd(effective(:,:,k),'econ');
    weights = F*V(:,1);
    precoder(:,k) = weights/norm(weights);
end
end
