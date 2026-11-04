function packed = muFrequencyPack(input)
[W,L,F,N] = size(input);
assert(mod(W,4)==0);
packed = reshape(permute(reshape(input,4,W/4,L,F,N),[2 3 1 4 5]),W/4,L,4*F,N);
end
