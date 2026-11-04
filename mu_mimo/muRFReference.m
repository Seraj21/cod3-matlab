function h = muRFReference(h,txPhase,rxPhase,frequency,practicalOffset,trueOffset,r)
% Mean RF phase is evaluated over the same half-CP FFT window as demodulation.
L = r.L;
Nfft = double(r.OFDM.Nfft);
start = [0 cumsum(double(r.OFDM.SymbolLengths(1:L-1)))];
phase = complex(zeros(1,L));
for l = 1:L
    first = start(l)+floor(.5*double(r.OFDM.CyclicPrefixLengths(l)))+1;
    tx = first+(0:Nfft-1);
    rx = tx+trueOffset;
    tx = min(numel(txPhase),tx);
    rx = min(numel(rxPhase),rx);
    rotation = exp(-1i*2*pi*frequency*(rx-1)/r.OFDM.SampleRate);
    phase(l) = mean(txPhase(tx).*rxPhase(rx).*rotation.');
end
step = 2*pi*(-r.K/2:r.K/2-1).'/Nfft;
h = double(h).*reshape(phase,1,L,1,1).*exp(1i*step*(practicalOffset-trueOffset));
end
