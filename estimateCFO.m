function frequency = estimateCFO(waveform,offset,ofdm,symbols)
% CP correlation on the latter half of each CP reduces multipath edge bias.
correlation = 0;
first = offset+1;
for symbol = 1:symbols
    prefixLength = double(ofdm.CyclicPrefixLengths(symbol));
    prefix = first+(floor(prefixLength/2):prefixLength-1);
    copy = prefix+double(ofdm.Nfft);
    correlation = correlation+sum(conj(waveform(prefix,:)).*waveform(copy,:),'all');
    first = first+double(ofdm.SymbolLengths(symbol));
end
frequency = angle(correlation)*double(ofdm.SampleRate)/(2*pi*double(ofdm.Nfft));
end
