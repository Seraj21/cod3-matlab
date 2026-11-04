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
P = numel(rays.Delays);
k = (-K/2:K/2-1).';
offset = 8;
output = complex(zeros(N,Nr));
if nargout>1
    starts = [0 cumsum(double(ofdm.SymbolLengths(1:L-1)))];
    % nrOFDMDemodulate starts its FFT halfway through the CP by default.
    starts = starts+floor(.5*double(ofdm.CyclicPrefixLengths(1:L)))+offset;
    centers = starts+(double(ofdm.Nfft)-1)/2;
    coefficients = complex(zeros(K,L,P,'single'));
    spatial = complex(zeros(P,Nr*Nt,'single'));
end
for path = 1:P
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
    if nargout>1
        spectrum = exp(-1i*2*pi*k*(integer+(0:16)-offset)/ofdm.Nfft)*kernel.';
        averageGain = sinc(frequency*double(ofdm.Nfft)/ofdm.SampleRate)/sinc(frequency/ofdm.SampleRate);
        phase = averageGain*exp(1i*2*pi*frequency*(channel.InitialTime+centers/ofdm.SampleRate));
        coefficients(:,:,path) = single(spectrum*phase);
        spatial(path,:) = reshape(single(rays.Receive(:,path)*rays.Transmit(:,path)'),1,[]);
    end
end
if nargout>1
    response = reshape(reshape(coefficients,K*L,P)*spatial,K,L,Nr,Nt);
end
end
