function code1233_link(M, numSlots, outputFolder)
% Single-layer OFDM link with TDL-C, perfect CSI, and CRC-verified payload.
arguments
    M (1,1) double {mustBeMember(M,[256 512])} = 256
    numSlots (1,1) double {mustBeInteger,mustBePositive} = 64
    outputFolder (1,1) string = "results"
end
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

carrier = nrCarrierConfig('NSizeGrid',66,'SubcarrierSpacing',60);
ofdm = nrOFDMInfo(carrier);
K = 12 * carrier.NSizeGrid;
L = carrier.SymbolsPerSlot;
slotSamples = sum(ofdm.SymbolLengths(1:L));
slotDuration = slotSamples / ofdm.SampleRate;
bandwidth = 50e6;
carrierFrequency = 28e9;
speedKmh = 30;
maxDoppler = (speedKmh / 3.6) * carrierFrequency / 299792458;
delaySpread = 300e-9;
targetRate = 0.5;
snrDb = 0:5:40;
bitsPerSymbol = log2(M);

% Reserve two symbols for control and transmit one-port PDSCH DM-RS.
pdsch = nrPDSCHConfig('PRBSet',0:carrier.NSizeGrid-1,'SymbolAllocation',[2 12], ...
    'MappingType','A','Modulation','256QAM','NumLayers',1);
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
dataIndices = nrPDSCHIndices(carrier,pdsch);
pilotIndices = nrPDSCHDMRSIndices(carrier,pdsch);
pilotSymbols = nrPDSCHDMRS(carrier,pdsch);
assert(isempty(intersect(dataIndices,pilotIndices)));
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
    'TransportBlockLength',payloadBitsPerSlot,'MaximumLDPCIterationCount',12);

configs = [1 1 1 1; 1 4 1 4; 4 1 4 1; 4 4 4 4; 16 16 4 4; 32 32 8 8];
configNames = {'SISO (1x1)','SIMO (1x4)','MISO (4x1)','MIMO (4x4)', ...
    'mMIMO (16x16 HBF)','mMIMO (32x32 HBF)'};
dimensions = [size(configs,1), numel(snrDb)];
bitErrors = zeros(dimensions);
blockErrors = zeros(dimensions);
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
        layerGrid(dataIndices) = qammod(coded,M,'InputType','bit','UnitAveragePower',true);
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
        for s = 1:numel(snrDb)
            noiseVariance = 10^(-snrDb(s)/10);
            receivedGrid = nrOFDMDemodulate(carrier,rx+sqrt(noiseVariance)*baseNoise);
            equalized = sum(conj(Hgrid).*receivedGrid,3)./gain;
            soft = qamdemod(equalized(dataIndices),M,'OutputType','approxllr', ...
                'UnitAveragePower',true,'NoiseVariance',1);
            weights = reshape(gain(dataIndices),1,[])/noiseVariance;
            soft = reshape(reshape(soft,bitsPerSymbol,[]).*weights,[],1);
            bitErrors(c,s) = bitErrors(c,s) + nnz((soft < 0) ~= logical(coded(:)));
            reset(decoder);
            [decoded,crcFailed] = decoder(max(-50,min(50,soft)),rateMatchMod,1,0);
            % Keep the CRC metric, with the known payload as a cross-check.
            blockErrors(c,s) = blockErrors(c,s) + ...
                double(crcFailed || any(decoded(:) ~= payload(:)));
        end
        if mod(slot,16) == 0 || slot == numSlots
            fprintf('  %d/%d slots\n',slot,numSlots);
        end
    end
end

totalBits = repmat(codedBitsPerSlot*numSlots,dimensions);
totalBlocks = repmat(numSlots,dimensions);
ber = bitErrors./totalBits;
bler = blockErrors./totalBlocks;
rawSE = (1-ber)*codedBitsPerSlot/(slotDuration*bandwidth);
goodputMbps = (1-bler)*payloadBitsPerSlot/slotDuration/1e6;
netSE = goodputMbps*1e6/bandwidth;
metrics = table(repelem(configNames(:),numel(snrDb)),repmat(snrDb(:),size(configs,1),1), ...
    reshape(bitErrors.',[],1),reshape(totalBits.',[],1), ...
    reshape(blockErrors.',[],1),reshape(totalBlocks.',[],1), ...
    reshape(ber.',[],1),reshape(bler.',[],1),reshape(rawSE.',[],1), ...
    reshape(goodputMbps.',[],1),reshape(netSE.',[],1), ...
    'VariableNames',{'Configuration','SNR_dB','BitErrors','TotalCodedBits','BlockErrors', ...
    'TotalBlocks','BER','BLER','RawCorrectBitSE','Goodput_Mbps','NetSE'});
writetable(metrics,fullfile(outputFolder,'metrics.csv'));
save(fullfile(outputFolder,'link_results.mat'),'M','numSlots','configs','configNames','snrDb', ...
    'ber','bler','rawSE','goodputMbps','netSE','bitErrors','totalBits','blockErrors','totalBlocks', ...
    'codedBitsPerSlot','payloadBitsPerSlot','slotDuration','bandwidth','maxDoppler','delaySpread');
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
