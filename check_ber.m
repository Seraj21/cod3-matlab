function check_ber
carrier = nrCarrierConfig('NSizeGrid',66,'SubcarrierSpacing',60);
ofdm = nrOFDMInfo(carrier);
L = carrier.SymbolsPerSlot;
K = carrier.NSizeGrid*12;
slotSamples = sum(ofdm.SymbolLengths(1:L));
pdsch = nrPDSCHConfig('PRBSet',0:65,'SymbolAllocation',[2 12], ...
    'MappingType','A','Modulation','256QAM','NumLayers',1);
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
dataIndices = nrPDSCHIndices(carrier,pdsch);
pilotIndices = nrPDSCHDMRSIndices(carrier,pdsch);
pilotSymbols = nrPDSCHDMRS(carrier,pdsch);
doppler = (30/3.6)*28e9/299792458;
scenarios = [300e-9 doppler; 300e-9 0; 30e-9 doppler; 30e-9 0];
snrDb = [20 35 40 Inf];
trials = 16;

for M = [256 512]
    bitsPerSymbol = log2(M);
    nbits = bitsPerSymbol*numel(dataIndices);
    for row = 1:size(scenarios,1)
        errors = zeros(size(snrDb));
        awgnErrors = errors;
        responseDelta = 0;
        signalErrorPower = 0;
        sampleCount = 0;
        for slot = 1:trials
            % Keep the data and base noise paired across scenarios.
            stream = RandStream('mt19937ar','Seed',10001+slot);
            bits = logical(randi(stream,[0 1],nbits,1));
            grid = complex(zeros(K,L));
            grid(dataIndices) = qammod(bits,M,'InputType','bit','UnitAveragePower',true);
            grid(pilotIndices) = pilotSymbols;
            tx = nrOFDMModulate(carrier,grid,'Windowing',0);
            loopback = nrOFDMDemodulate(carrier,tx);
            assert(max(abs(loopback(:)-grid(:))) < 1e-10);
            noise = (randn(stream,slotSamples,1)+1i*randn(stream,slotSamples,1)) / ...
                sqrt(2*double(ofdm.Nfft));
            frequencyNoise = nrOFDMDemodulate(carrier,noise);
            assert(abs(mean(abs(frequencyNoise(dataIndices)).^2)-1) < .08);

            ch = nrTDLChannel('DelayProfile','TDL-C','DelaySpread',scenarios(row,1), ...
                'MaximumDopplerShift',scenarios(row,2),'SampleRate',ofdm.SampleRate, ...
                'PathGainSampleRate','auto','NumTransmitAntennas',1,'NumReceiveAntennas',1, ...
                'NormalizeChannelOutputs',false,'RandomStream','mt19937ar with seed', ...
                'Seed',31000+slot,'ChannelResponseOutput','ofdm-response');
            delay = info(ch).MaximumChannelDelay;
            probe = clone(ch);
            probe.ChannelFiltering = false;
            probe.NumTimeSamples = slotSamples + delay;
            [Hprobe,tProbe] = probe(carrier);
            [waveform,H,t] = ch([tx; zeros(delay,1)],carrier);
            assert(tProbe == t);
            relativeDifference = norm(Hprobe(:)-H(:))/norm(H(:));
            responseDelta = max(responseDelta,relativeDifference);
            % Normalize the arbitrary SISO phase factors away. The same
            % effective diagonal is used by the full single-layer receiver.
            noiseless = nrOFDMDemodulate(carrier,waveform(t+(1:slotSamples)));
            estimated = noiseless(dataIndices)./H(dataIndices);
            signalErrorPower = signalErrorPower + sum(abs(estimated-grid(dataIndices)).^2);
            sampleCount = sampleCount + numel(dataIndices);
            for j = 1:numel(snrDb)
                scale = 10^(-snrDb(j)/20);
                data = (noiseless(dataIndices)+scale*frequencyNoise(dataIndices))./Hprobe(dataIndices);
                hard = logical(qamdemod(data,M,'OutputType','bit','UnitAveragePower',true));
                soft = qamdemod(data,M,'OutputType','approxllr','UnitAveragePower',true);
                assert(isequal(hard,soft < 0));
                errors(j) = errors(j)+nnz(hard ~= bits);
                awgn = qamdemod(grid(dataIndices)+scale*frequencyNoise(dataIndices),M, ...
                    'OutputType','bit','UnitAveragePower',true);
                awgnErrors(j) = awgnErrors(j)+nnz(logical(awgn) ~= bits);
            end
        end
        fprintf('QAM=%d spread_ns=%.0f doppler=%.1f probe_diff=%.3g evm_rms=%.6g trials=%d\n', ...
            M,scenarios(row,1)*1e9,scenarios(row,2),responseDelta,sqrt(signalErrorPower/sampleCount),trials);
        fprintf('SNR [20,35,40,Inf] TDL BER: %.9g %.9g %.9g %.9g\n',errors/(trials*nbits));
        fprintf('SNR [20,35,40,Inf] AWGN BER: %.9g %.9g %.9g %.9g\n',awgnErrors/(trials*nbits));
    end
end
end
