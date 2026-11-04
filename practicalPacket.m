function packet = practicalPacket(carrier,layerGrid,settings,profile,spread,speed,seed)
% Two actual 8-port CSI-RS transmissions precede one data slot.
ofdm = nrOFDMInfo(carrier);
K = size(layerGrid,1);
L = size(layerGrid,2);
slotSamples = sum(ofdm.SymbolLengths(1:L));
slotDuration = slotSamples/ofdm.SampleRate;
doppler = speed/3.6*settings.CarrierFrequency/299792458;
channel = practicalChannel(settings.ArraySize,ofdm.SampleRate,profile,spread,doppler,seed);
channelInfo = info(channel);
tail = max(channelInfo.MaximumChannelDelay,ceil(12e-6*ofdm.SampleRate));
[banks,WRF] = hybridCodebook(settings.ArraySize/8,8,settings.PhaseBits);
rxRoot = chol(WRF'*WRF);
receiver = WRF/rxRoot;
csirs = nrCSIRSConfig('CSIRSType','nzp','RowNumber',6,'Density','one', ...
    'SubcarrierLocations',[0 2 4 6],'SymbolLocations',0,'NumRB',carrier.NSizeGrid);
indices = nrCSIRSIndices(carrier,csirs);
symbols = nrCSIRS(carrier,csirs)/sqrt(2);
soundGrid = nrResourceGrid(carrier,8);
soundGrid(indices) = symbols;
scores = zeros(1,numel(banks));
estimates = cell(1,numel(banks));
noise = RandStream('mt19937ar','Seed',seed+100000);
for bank = 1:numel(banks)
    RF = banks{bank};
    grid = reshape(reshape(soundGrid,K*L,8)*RF.',K,L,settings.ArraySize);
    tx = nrOFDMModulate(carrier,grid,'Windowing',0);
    [tx,~,~] = radioImpairments(tx,settings,seed+200000+bank,true,0);
    release(channel);
    channel.InitialTime = (bank-1)*slotDuration;
    [rx,~,~] = channel([tx;zeros(tail,settings.ArraySize)],carrier);
    [rx,~,~] = radioImpairments(rx,settings,seed+300000+bank,false,channel.InitialTime);
    % Antenna noise is white. Equivalent noise after explicit RF whitening is white.
    rx = rx*conj(receiver);
    referenceNoise = complex(randn(noise,size(rx)),randn(noise,size(rx))) ...
        /sqrt(2*double(ofdm.Nfft)*10^(settings.SoundingSNR/10));
    rx = rx+referenceNoise;
    [offset,mag] = nrTimingEstimate(carrier,rx,indices,symbols);
    [~,peak] = max(sum(abs(mag(1:min(tail+1,size(mag,1)),:)).^2,2));
    offset = peak-1;
    frequency = estimateCFO(rx,offset,ofdm,L);
    rx = rx.*exp(-1i*2*pi*frequency*(0:size(rx,1)-1).'/ofdm.SampleRate);
    received = nrOFDMDemodulate(carrier,rx(offset+(1:slotSamples),:));
    estimate = nrChannelEstimate(carrier,received,indices,symbols,'CDMLengths',[2 1]);
    % Estimate is referred to the whitened receiver. Undo only that digital stage.
    effective = permute(mean(estimate,2),[3 4 1 2]);
    estimates{bank} = pagemtimes(rxRoot',double(effective));
    scores(bank) = mean(abs(effective).^2,'all');
end
[~,selected] = max(scores);
weights = hybridPrecoder(banks{selected},WRF,estimates{selected},settings.PRGWidth);
antennaGrid = reshape(single(layerGrid),K,L,1).*reshape(single(weights.Precoder.'),K,1,[]);
tx = nrOFDMModulate(carrier,antennaGrid,'Windowing',0);
[distorted,txPhase,paGain] = radioImpairments(tx,settings,seed+400000,true,0);
release(channel);
channel.InitialTime = numel(banks)*slotDuration;
[rx,response,offset] = channel([distorted;zeros(tail,settings.ArraySize)],carrier);
[rx,rxPhase,~] = radioImpairments(rx,settings,seed+500000,false,channel.InitialTime);
packet.AntennaPower = mean(abs(nrOFDMDemodulate(carrier,rx(offset+(1:slotSamples),:))).^2,'all');
packet.Waveform = double(rx*conj(weights.Receiver));
H = permute(response(:,1:L,:,:),[3 4 1 2]);
linearPrecoder = weights.Precoder.*paGain.';
projected = pagemtimes(weights.Receiver',H);
H = pagemtimes(projected,reshape(linearPrecoder,settings.ArraySize,1,K,1));
H = double(reshape(permute(H,[3 4 1 2]),K,L,[]));
% The oracle includes the mean RF phase over each useful OFDM symbol.
centers = zeros(1,L);
commonPhase = zeros(1,L);
first = 1;
for symbol = 1:L
    cp = double(ofdm.CyclicPrefixLengths(symbol));
    samples = first+cp+(0:double(ofdm.Nfft)-1);
    centers(symbol) = mean(samples)-1+offset;
    txSamples = min(numel(txPhase),samples);
    rxSamples = min(numel(rxPhase),samples+offset);
    commonPhase(symbol) = mean(txPhase(txSamples).*rxPhase(rxSamples));
    first = first+double(ofdm.SymbolLengths(symbol));
end
packet.Channel = H.*reshape(commonPhase,1,L,1);
packet.SymbolCenters = centers;
packet.TimingOffset = double(offset);
packet.Weights = weights;
packet.BeamScores = scores;
packet.SelectedBank = selected;
packet.TxPower = mean(sum(abs(antennaGrid).^2,3),'all');
packet.PAOutputPower = mean(abs(distorted).^2,'all')/mean(abs(tx).^2,'all');
packet.SoundingDuration = numel(banks)*slotDuration;
end
