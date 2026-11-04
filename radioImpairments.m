function [output,phase,gain] = radioImpairments(input,settings,seed,transmit,time)
% Shared local oscillator per array. PA settings are explicit experiment inputs.
N = size(input,1);
oscillator = comm.PhaseNoise('Level',settings.PhaseNoiseLevel, ...
    'FrequencyOffset',settings.PhaseNoiseOffset,'SampleRate',settings.SampleRate, ...
    'RandomStream','mt19937ar with seed','Seed',seed);
phase = oscillator(complex(ones(N,1)));
gain = ones(1,size(input,2));
output = input;
if transmit
    saturation = sqrt(mean(abs(input).^2,'all'))*10^(settings.InputBackoff/20);
    saturation = max(saturation,eps);
    ratio = abs(input)/saturation;
    output = input./(1+ratio.^6).^(1/6);
    output = output.*exp(1i*settings.AMPMDegrees*pi/180*ratio.^2./(1+ratio.^2));
    gain = sum(conj(input).*output,1)./max(sum(abs(input).^2,1),eps);
else
    phase = phase.*exp(1i*2*pi*settings.FrequencyOffset*((0:N-1).'/settings.SampleRate+time));
end
output = output.*phase;
end
