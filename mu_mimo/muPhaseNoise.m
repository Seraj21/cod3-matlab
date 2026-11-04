function phase = muPhaseNoise(count,settings,seed)
% Reuse the phase-noise filter, while preserving caller RNG state and seeds.
persistent keys oscillators streams
key = sprintf('%.14g,',[settings.SampleRate settings.PhaseNoiseLevel settings.PhaseNoiseOffset]);
index = find(strcmp(keys,key),1);
original = RandStream.getGlobalStream;
restore = onCleanup(@()RandStream.setGlobalStream(original)); %#ok<NASGU>
if isempty(index)
    stream = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(stream);
    oscillator = comm.PhaseNoise('Level',settings.PhaseNoiseLevel, ...
        'FrequencyOffset',settings.PhaseNoiseOffset,'SampleRate',settings.SampleRate, ...
        'RandomStream','Global stream');
    index = numel(keys)+1;
    keys{index} = key;
    streams{index} = stream;
    oscillators{index} = oscillator;
else
    stream = streams{index};
    reset(stream,seed);
    RandStream.setGlobalStream(stream);
    oscillator = oscillators{index};
    reset(oscillator);
end
phase = oscillator(complex(ones(count,1)));
end
