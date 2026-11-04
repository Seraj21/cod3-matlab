function [output,response,offset] = applyMIMOChannel(channel,input,carrier)
if ~isstruct(channel)
    [output,response,offset] = channel(input,carrier);
    return
end
% External ray channel with a 17-tap windowed-sinc fractional-delay filter.
rays = channel.Rays;
N = size(input,1);
Nt = size(input,2);
Nr = size(rays.Receive,1);
ofdm = nrOFDMInfo(carrier);
K = 12*carrier.NSizeGrid;
L = carrier.SymbolsPerSlot;
k = (-K/2:K/2-1).';
offset = 8;
output = complex(zeros(N,Nr));
response = complex(zeros(K,L,Nr,Nt,'single'));
starts = [0 cumsum(double(ofdm.SymbolLengths(1:L-1)))];
centers = starts+double(ofdm.CyclicPrefixLengths(1:L))+double(ofdm.Nfft)/2+offset;
for path = 1:numel(rays.Delays)
    if norm(rays.Receive(:,path))==0
        continue
    end
    samples = rays.Delays(path)*ofdm.SampleRate;
    integer = floor(samples);
    fractional = samples-integer;
    x = (0:16)-8-fractional;
    kernel = ones(size(x));
    nonzero = abs(x)>1e-12;
    kernel(nonzero) = sin(pi*x(nonzero))./(pi*x(nonzero));
    kernel = kernel.*(.5-.5*cos(2*pi*(0:16)/16));
    kernel = kernel/sum(kernel);
    projected = input*conj(rays.Transmit(:,path));
    delayed = filter(kernel,1,projected);
    delayed = [zeros(integer,1);delayed(1:N-integer)];
    frequency = channel.Doppler*rays.DopplerFactor(path);
    rotation = exp(1i*2*pi*frequency*(channel.InitialTime+(0:N-1).'/ofdm.SampleRate));
    output = output+(delayed.*rotation)*rays.Receive(:,path).';
    spectrum = exp(-1i*2*pi*k*(integer+(0:16)-offset)/ofdm.Nfft)*kernel.';
    phase = exp(1i*2*pi*frequency*(channel.InitialTime+centers/ofdm.SampleRate));
    spatial = rays.Receive(:,path)*rays.Transmit(:,path)';
    component = reshape(spectrum*phase,K,L,1,1).*reshape(spatial,1,1,Nr,Nt);
    response = response+single(component);
end
end
