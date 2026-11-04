function output = muCNNRestore(packed,meta,scale)
W = meta.W;
L = meta.L;
G = meta.G;
C = meta.Coefficients;
N = G*meta.Blocks;
y = reshape(permute(reshape(packed,W/4,L,4,2*C,N),[3 1 2 4 5]),W,L,2*C,N);
y = complex(y(:,:,1:C,:),y(:,:,C+(1:C),:));
output = complex(zeros(meta.K,L,meta.R,meta.U));
for b = 1:meta.Blocks
    rows = (b-1)*meta.GroupSize+(1:meta.GroupSize);
    indices = (b-1)*G+(1:G);
    p = reshape(y(:,:,:,indices),W,L,meta.GroupSize,meta.U,G);
    amplitude = permute(scale(:,rows,:),[4 5 2 3 1]);
    p = p.*amplitude;
    output(:,:,rows,:) = reshape(permute(p,[1 5 2 3 4]),meta.K,L,meta.GroupSize,meta.U);
end
end
