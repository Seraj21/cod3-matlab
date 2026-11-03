function code1233_link(M, numSlots, outputFolder)
% Post-LDPC BER for a high-mobility, long-delay TDL-C channel with perfect CSI.
arguments
    M (1,1) double {mustBeMember(M,[256 512])} = 256
    numSlots (1,1) double {mustBeInteger,mustBePositive} = 64
    outputFolder (1,1) string = "results"
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
snrDb = 0:5:40;
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
[~,pilotSymbolIndex] = ind2sub([K L],pilotIndices);
[~,dataSymbolIndex] = ind2sub([K L],dataIndices);
pilotOFDMSymbols = unique(pilotSymbolIndex);
assert(numel(pilotOFDMSymbols) >= 2,'Expected DM-RS in at least two OFDM symbols.');
pilotGroups = pilotSymbolIndex == pilotOFDMSymbols.';
querySymbols = min(pilotOFDMSymbols(end),max(pilotOFDMSymbols(1),dataSymbolIndex));
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
dimensions = [size(configs,1), numel(snrDb)];
postBitErrors = zeros(dimensions);
for c = 1:size(configs,1)
    Nt = configs(c,1);
    Nr = configs(c,2);
    fprintf('%d-QAM | %s | %d slots\n',M,configNames{c},numSlots);
    % Each configuration and slot has the same channel/noise seeds in both
    % modulation jobs. Within a slot, all SNR points see the same waveform.
    payloadStream = RandStream('mt19937ar','Seed',10000+c);
    noiseStream = RandStream('mt19937ar','Seed',20000+c);

    probe = makeChannel(Nt,Nr,ofdm.SampleRate,delaySpread,maxDoppler);
    probe.ChannelFiltering = false;
    probe.ChannelResponseOutput = 'ofdm-response';
    channel = makeChannel(Nt,Nr,ofdm.SampleRate,delaySpread,maxDoppler);
    channelInfo = info(channel);
    tailSamples = channelInfo.MaximumChannelDelay;
    probe.NumTimeSamples = slotSamples + tailSamples;
    for slot = 1:numSlots
        seed = 30000 + 1000*c + slot;
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
        antennaGrid = reshape(layerGrid,K,L,1) .* reshape(precoder.',K,1,Nt);
        tx = nrOFDMModulate(carrier,antennaGrid,'Windowing',0);
        assert(size(tx,1) == slotSamples);
        rx = channel([tx; zeros(tailSamples,Nt)]);
        rx = rx(timingOffset+(1:slotSamples),:) * conj(receiver);

        % Receiver is an orthonormal basis of the analog combiner span.
        % Its noise remains white. Use the actual per-symbol channel for
        % normalized single-layer combining, while the precoder is cached.
        projected = pagemtimes(receiver','none',H,'none');
        Hsingle = pagemtimes(projected,reshape(precoder,Nt,1,K,1));
        Hgrid = reshape(permute(Hsingle,[3 4 1 2]),K,L,[]);
        gain = max(sum(abs(Hgrid).^2,3),eps);
        % Scale time-domain AWGN to unit noise variance per subcarrier at
        % 0 dB. Keep the same base noise realization across the SNR sweep.
        baseNoise = (randn(noiseStream,size(rx)) + 1i*randn(noiseStream,size(rx))) ...
            / sqrt(2*double(ofdm.Nfft));
        % OFDM demodulation and linear combining can be reused across SNRs.
        signalGrid = nrOFDMDemodulate(carrier,rx);
        noiseGrid = nrOFDMDemodulate(carrier,baseNoise);
        equalizedSignal = sum(conj(Hgrid).*signalGrid,3)./gain;
        equalizedNoise = sum(conj(Hgrid).*noiseGrid,3)./gain;
        pilotGain = gain(pilotIndices);
        weightedPilotGroups = double(pilotGroups).*pilotGain;
        groupGain = max(sum(weightedPilotGroups,1).',eps);
        groupSize = sum(pilotGroups,1).';
        deinterleaved = zeros(codedBitsPerSlot,1);
        for s = 1:numel(snrDb)
            noiseVariance = 10^(-snrDb(s)/10);
            equalized = equalizedSignal + sqrt(noiseVariance)*equalizedNoise;
            pilotErrorPower = abs(equalized(pilotIndices)-pilotSymbols).^2;
            % Estimate equalized residual interference from known DM-RS.
            % Remove the expected AWGN contribution before interpolation.
            pilotResidual = max(0,(weightedPilotGroups.'*pilotErrorPower - ...
                noiseVariance*groupSize)./groupGain);
            residual = interp1(pilotOFDMSymbols,pilotResidual,querySymbols,'linear');
            effectiveVariance = noiseVariance./gain(dataIndices) + residual;
            soft = qamdemod(equalized(dataIndices),M,'OutputType','approxllr', ...
                'UnitAveragePower',true,'NoiseVariance',1);
            weights = 1./reshape(effectiveVariance,1,[]);
            deinterleaved(bitOrder) = reshape(reshape(soft,bitsPerSymbol,[]).*weights,[],1);
            reset(decoder);
            decoded = decoder(max(-50,min(50,deinterleaved)),rateMatchMod,1,0);
            payloadErrors = nnz(decoded(:) ~= payload(:));
            postBitErrors(c,s) = postBitErrors(c,s) + payloadErrors;
        end
        if mod(slot,16) == 0 || slot == numSlots
            fprintf('  %d/%d slots\n',slot,numSlots);
        end
    end
end

totalPayloadBits = repmat(payloadBitsPerSlot*numSlots,dimensions);
postBer = postBitErrors./totalPayloadBits;
metrics = table(repelem(configNames(:),numel(snrDb)),repmat(snrDb(:),size(configs,1),1), ...
    reshape(postBer.',[],1),reshape(postBitErrors.',[],1),reshape(totalPayloadBits.',[],1), ...
    'VariableNames',{'Configuration','SNR_dB','PostLDPC_BER','PostLDPC_BitErrors','TotalPayloadBits'});
writetable(metrics,fullfile(outputFolder,'post_ldpc_ber.csv'));
save(fullfile(outputFolder,'post_ldpc_ber.mat'),'M','numSlots','configs','configNames','snrDb', ...
    'postBer','postBitErrors','totalPayloadBits','payloadBitsPerSlot','maxDoppler','delaySpread', ...
    'speedKmh','carrierFrequency','targetRate','ofdm');
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
