function channel = practicalChannel(arraySize,sampleRate,profile,delaySpread,doppler,seed,settings)
if profile=="DeepMIMO"
    persistent sourceFile source
    if isempty(sourceFile) || sourceFile~=settings.RayFile
        source = load(settings.RayFile);
        sourceFile = settings.RayFile;
    end
    index = settings.RayIndex;
    assert(index<=size(source.delays,1),"External ray position index exceeds dataset.");
    receive = source.(sprintf('receive%d',arraySize));
    transmit = source.(sprintf('transmit%d',arraySize));
    rays = struct('Receive',reshape(receive(index,:,:),arraySize,[]), ...
        'Transmit',reshape(transmit(index,:,:),arraySize,[]), ...
        'Delays',source.delays(index,:),'DopplerFactor',source.doppler_factor(index,:));
    channel = struct('Rays',rays,'InitialTime',0,'Doppler',doppler);
    return
elseif startsWith(profile,"CDL")
    channel = nrCDLChannel('DelayProfile',char(profile),'DelaySpread',delaySpread, ...
        'MaximumDopplerShift',doppler,'CarrierFrequency',28e9,'SampleRate',sampleRate, ...
        'NormalizePathGains',true,'NormalizeChannelOutputs',false, ...
        'RandomStream','mt19937ar with seed','Seed',seed);
    channel.TransmitAntennaArray.Size = [arraySize/8 8 1 1 1];
    channel.ReceiveAntennaArray.Size = [arraySize/8 8 1 1 1];
    channel.TransmitAntennaArray.ElementSpacing = [.5 .5 1 1];
    channel.ReceiveAntennaArray.ElementSpacing = [.5 .5 1 1];
    channel.TransmitAntennaArray.Element = 'isotropic';
    channel.ReceiveAntennaArray.Element = 'isotropic';
else
    channel = nrTDLChannel('DelayProfile',char(profile),'DelaySpread',delaySpread, ...
        'MaximumDopplerShift',doppler,'SampleRate',sampleRate,'PathGainSampleRate','auto', ...
        'NumTransmitAntennas',arraySize,'NumReceiveAntennas',arraySize, ...
        'NormalizePathGains',true,'NormalizeChannelOutputs',false, ...
        'RandomStream','mt19937ar with seed','Seed',seed);
end
channel.ChannelResponseOutput = 'ofdm-response';
end
