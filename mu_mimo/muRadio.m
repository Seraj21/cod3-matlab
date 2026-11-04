function [output,phase,gain] = muRadio(input,cfg,seed,transmit,time)
N = size(input,1);
phase = muPhaseNoise(N,cfg,seed);
gain = ones(1,size(input,2));
output = input;
if transmit
    active = any(abs(input)>0,2);
    if any(active)
        saturation = sqrt(mean(abs(input(active,:)).^2,'all'))*10^(cfg.InputBackoff/20);
    else
        saturation = eps;
    end
    saturation = max(saturation,eps);
    ratio = abs(input)/saturation;
    output = input./(1+ratio.^6).^(1/6);
    output = output.*exp(1i*cfg.AMPMDegrees*pi/180*ratio.^2./(1+ratio.^2));
    gain = sum(conj(input).*output,1)./max(sum(abs(input).^2,1),eps);
else
    phase = phase.*exp(1i*2*pi*cfg.FrequencyOffset*((0:N-1).'/cfg.SampleRate+time));
end
output = output.*phase;
end
