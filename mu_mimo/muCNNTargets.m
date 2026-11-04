function packed = muCNNTargets(error,scale,meta)
W = meta.W;
L = meta.L;
G = meta.G;
C = meta.Coefficients;
parts = cell(meta.Blocks,1);
for b = 1:meta.Blocks
    rows = (b-1)*meta.GroupSize+(1:meta.GroupSize);
    y = permute(reshape(error(:,:,rows,:),W,G,L,meta.GroupSize,meta.U),[1 3 4 5 2]);
    amplitude = permute(scale(:,rows,:),[4 5 2 3 1]);
    y = reshape(y./amplitude,W,L,C,G);
    parts{b} = muFrequencyPack(single(cat(3,real(y),imag(y))));
end
packed = cat(4,parts{:});
end
