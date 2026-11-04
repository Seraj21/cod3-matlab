function value=v2ParameterRestore(packed,meta)
W=meta.W;L=meta.L;C=meta.Coefficients;G=meta.G;
p=reshape(permute(reshape(packed,W/4,L,4,C,G),[3 1 2 4 5]),W,L,C,G);
p=reshape(p,W,L,meta.GroupSize,meta.U,G);
value=reshape(permute(p,[1 5 2 3 4]),meta.K,L,meta.GroupSize,meta.U);
end
