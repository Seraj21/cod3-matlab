function [X,meta] = muCNNFeatures(state,layouts,r,groupSize)
[K,L,R,U] = size(state.LS);
W = layouts{1}.W;
G = K/W;
assert(mod(R,groupSize)==0);
blocks = R/groupSize;
C = groupSize*U;
features = cell(blocks,1);
for b = 1:blocks
    rows = (b-1)*groupSize+(1:groupSize);
    scale = permute(state.Scale(:,rows,:),[4 5 2 3 1]);
    ls = toPatches(state.LS(:,:,rows,:),W)./scale;
    wiener = toPatches(state.Wiener(:,:,rows,:),W)./scale;
    sparse = toPatches(state.Pilot(:,:,rows,:),W)./scale;
    relative = permute(state.RelativeNoise(:,rows,:),[4 5 2 3 1]);
    relative = repmat(log10(max(1e-6,relative)),W,L,1,1,1);
    mask = false(W,L,groupSize,U,G);
    for u = 1:U
        mask(:,:,:,u,:) = repmat(layouts{u}.Mask,1,1,groupSize,1,G);
    end
    shape = @(a)reshape(a,W,L,C,G);
    time = repmat(layouts{1}.TimeCoordinate,1,1,1,G);
    frequency = repmat(layouts{1}.FrequencyCoordinate,1,1,1,G);
    data = cat(3,shape(real(ls)),shape(imag(ls)),shape(real(wiener)),shape(imag(wiener)), ...
        shape(real(sparse)),shape(imag(sparse)),shape(mask),shape(relative),time,frequency);
    features{b} = muFrequencyPack(single(data));
end
X = cat(4,features{:});
meta = struct('K',K,'L',L,'R',R,'U',U,'W',W,'G',G,'GroupSize',groupSize,'Blocks',blocks,'Coefficients',C);
end
function p = toPatches(h,W)
[K,L,R,U] = size(h);
p = permute(reshape(h,W,K/W,L,R,U),[1 3 4 5 2]);
end
