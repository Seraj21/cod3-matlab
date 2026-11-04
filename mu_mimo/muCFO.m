function frequency = muCFO(waveform,offset,ofdm,symbols)
starts = [0 cumsum(double(ofdm.SymbolLengths))];
correlation = 0;
for symbol = symbols(:).'
    cp = double(ofdm.CyclicPrefixLengths(symbol));
    prefix = offset+starts(symbol)+(floor(cp/2):cp-1)+1;
    copy = prefix+double(ofdm.Nfft);
    correlation = correlation+sum(conj(waveform(prefix,:)).*waveform(copy,:),'all');
end
frequency = angle(correlation)*ofdm.SampleRate/(2*pi*double(ofdm.Nfft));
end
