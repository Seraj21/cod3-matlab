function variance = muGuardNoise(waveform,r,tail)
% The late guard follows the maximum UL channel tail and precedes DL input.
guard=5;if isfield(r,'GuardSymbols'),guard=r.GuardSymbols;end
first = sum(r.OFDM.SymbolLengths(1:guard-1))+tail+1;
last = sum(r.OFDM.SymbolLengths(1:guard));
assert(last-first>=64,'Guard is too short for an independent noise measurement.');
variance = max(eps,double(r.OFDM.Nfft)*mean(abs(waveform(first:last,:)).^2,'all'));
end
