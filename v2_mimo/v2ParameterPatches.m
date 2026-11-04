function packed=v2ParameterPatches(value,meta)
% Pack a real-valued channel parameter using the same bijection as the CNN.
W=meta.W;L=meta.L;G=meta.G;R=meta.GroupSize;U=meta.U;
p=permute(reshape(value,W,G,L,R,U),[1 3 4 5 2]);
packed=muFrequencyPack(single(reshape(p,W,L,R*U,G)));
end
