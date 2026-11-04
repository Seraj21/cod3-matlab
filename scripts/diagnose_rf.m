function diagnose_rf(rayFile,outputFolder,arraySize)
% Diagnostic ablation on preidentified anomalous rays, with an oracle receiver.
% This diagnostic does not train, select, or tune the LS/LMMSE/CNN estimators.
if ~isfolder(outputFolder), mkdir(outputFolder); end
carrier = nrCarrierConfig('NSizeGrid',66,'SubcarrierSpacing',60,'CyclicPrefix','extended');
ofdm = nrOFDMInfo(carrier);
K = 12*carrier.NSizeGrid;
L = carrier.SymbolsPerSlot;
slotSamples = sum(ofdm.SymbolLengths(1:L));
pdsch = nrPDSCHConfig('PRBSet',0:65,'SymbolAllocation',[2 L-2], ...
    'MappingType','A','Modulation','256QAM','NumLayers',1,'EnablePTRS',true);
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
pdsch.DMRS.DMRSAdditionalPosition = 3;
pdsch.PTRS.TimeDensity = 1;
pdsch.PTRS.FrequencyDensity = 2;
data = nrPDSCHIndices(carrier,pdsch);
pilots = nrPDSCHDMRSIndices(carrier,pdsch);
symbols = nrPDSCHDMRS(carrier,pdsch);
ptrs = nrPDSCHPTRSIndices(carrier,pdsch);
ptrsSymbols = nrPDSCHPTRS(carrier,pdsch);
G = numel(data)*9;
A = 8*floor(.5*G/8);
codec = ExperimentalLDPC(A,G,9,.5);
orderStream = RandStream('mt19937ar','Seed',1729);
order = randperm(orderStream,G).';
settings = struct('ArraySize',arraySize,'CarrierFrequency',28e9,'PhaseBits',4, ...
    'PRGWidth',24,'SoundingSNR',15,'SampleRate',ofdm.SampleRate, ...
    'PhaseNoiseLevel',[-85 -100 -115],'PhaseNoiseOffset',[1e4 1e5 1e6], ...
    'InputBackoff',10,'AMPMDegrees',3,'FrequencyOffset',2800,'RayFile',string(rayFile));
results = [];
for ray = [1 85 94]
    batch = floor((ray-1)/32);
    trial = mod(ray-1,32)+1;
    offsetSeed = 85000+1000*batch;
    c = 6+(arraySize==64);
    seed = 30000+1000*c+offsetSeed+trial;
    stream = RandStream('mt19937ar','Seed',10000+c+offsetSeed);
    for j = 1:trial
        payload = int8(randi(stream,[0 1],A,1));
    end
    coded = codec.encode(payload);
    grid = complex(zeros(K,L));
    grid(data) = qammod(coded(order),512,'InputType','bit','UnitAveragePower',true);
    grid(pilots) = symbols;
    grid(ptrs) = ptrsSymbols;
    for condition = 1:4
        s = settings;
        s.RayIndex = ray;
        if condition<=2
            s.PhaseNoiseLevel = [-300 -300 -300];
            s.InputBackoff = 80;
            s.AMPMDegrees = 0;
            s.FrequencyOffset = 0;
        end
        speed = 120*ismember(condition,[2 4]);
        packet = practicalPacket(carrier,grid,s,"DeepMIMO",1e-6,speed,seed);
        noiseStream = RandStream('mt19937ar','Seed',20000+c+offsetSeed);
        for j = 1:trial
            baseNoise = complex(randn(noiseStream,size(packet.Waveform)), ...
                randn(noiseStream,size(packet.Waveform)))/sqrt(2*double(ofdm.Nfft));
        end
        nominalNoise = .01;
        rx = packet.Waveform+sqrt(nominalNoise)*baseNoise;
        offset = packet.TimingOffset;
        frequency = estimateCFO(rx,offset,ofdm,L);
        rx = rx.*exp(-1i*2*pi*frequency*(0:size(rx,1)-1).'/ofdm.SampleRate);
        observed = nrOFDMDemodulate(carrier,rx(offset+(1:slotSamples),:));
        h = receiverFrame(packet.Channel,2*pi*(-K/2:K/2-1).'/double(ofdm.Nfft), ...
            packet.SymbolCenters,frequency,offset,offset,ofdm.SampleRate);
        gain = max(sum(abs(h).^2,3),eps);
        eq = sum(conj(h).*observed,3)./gain;
        [~,times] = ind2sub([K L],ptrs);
        phasor = accumarray(times,eq(ptrs).*conj(ptrsSymbols),[L 1]);
        valid = find(abs(phasor)>0);
        phases = interp1(valid,unwrap(angle(phasor(valid))), ...
            min(valid(end),max(valid(1),(1:L).')),'linear');
        eq = eq.*exp(-1i*phases.');
        [~,pilotTimes] = ind2sub([K L],pilots);
        uniqueTimes = unique(pilotTimes);
        pilotLS = reshape(observed,K*L,[]);
        pilotLS = pilotLS(pilots,:)./symbols;
        samples = reshape(pilotLS,12,K/24,4,[]);
        error = samples(2:end-1,:,:,:)-(samples(1:end-2,:,:,:)+samples(3:end,:,:,:))/2;
        nVar = max(nominalNoise,median(abs(error).^2,'all')/(1.5*log(2)));
        group = double(pilotTimes==uniqueTimes.');
        interp = interp1(uniqueTimes,eye(4),min(uniqueTimes(end),max(uniqueTimes(1),(1:L).')),'linear');
        pg = gain(pilots);
        pe = abs(eq(pilots)-symbols).^2;
        residual = max(0,(group.'*(pg.*pe)-nVar*sum(group,1).')./max(group.'*pg,eps));
        residual = interp*residual;
        [~,dataTimes] = ind2sub([K L],data);
        variance = nVar./gain(data)+residual(dataTimes);
        soft = qamdemod(eq(data),512,'OutputType','approxllr', ...
            'UnitAveragePower',true,'NoiseVariance',1);
        soft = reshape(reshape(soft,9,[])./reshape(variance,1,[]),[],1);
        llr = zeros(G,1);
        llr(order) = soft;
        decoded = codec.decode(max(-50,min(50,llr)));
        evm = sqrt(mean(abs(eq(data)-grid(data)).^2)/mean(abs(grid(data)).^2));
        symbolPower = mean(gain,1);
        results = [results;arraySize ray condition speed 20 nnz(decoded~=payload) A evm ...
            min(symbolPower) median(symbolPower) max(symbolPower) frequency nVar]; %#ok<AGROW>
        fprintf('Array %d ray %d condition %d errors %d EVM %.5f\n', ...
            arraySize,ray,condition,nnz(decoded~=payload),evm);
    end
end
results = array2table(results,'VariableNames',{'Array','RayIndex','Condition', ...
    'Speed_kmh','SNR_dB','PayloadBitErrors','PayloadBits','DataEVM', ...
    'MinSymbolChannelPower','MedianSymbolChannelPower','MaxSymbolChannelPower','EstimatedCFO','NoiseEstimate'});
writetable(results,fullfile(outputFolder,'rf_ablation.csv'));
end
