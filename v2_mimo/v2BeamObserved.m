function beam=v2BeamObserved(physical,r)
Q=fft(eye(4))/2;
temporary=reshape(physical,r.K,r.L,4,4);
temporary=permute(temporary,[1 2 4 3]);
temporary=reshape(reshape(temporary,[],4)*conj(Q),r.K,r.L,4,4);
beam=reshape(permute(temporary,[1 2 4 3]),r.K,r.L,16);
end
