function weights = hybridPrecoder(FRF,WRF,measuredChannel,groupWidth)
% measuredChannel is WRF' * H * FRF with dimensions RF_rx by RF_tx by K.
% A single digital transmit layer is optimized with fixed total radiated power.
txRoot = chol(FRF'*FRF);
rxRoot = chol(WRF'*WRF);
weights.FRF = FRF;
weights.WRF = WRF;
weights.WBB = rxRoot\eye(size(rxRoot));
weights.Receiver = WRF*weights.WBB;
whitened = pagemtimes(rxRoot'\eye(size(rxRoot)),measuredChannel);
whitened = pagemtimes(whitened,txRoot\eye(size(txRoot)));
K = size(measuredChannel,3);
weights.FBB = complex(zeros(size(FRF,2),K));
weights.Precoder = complex(zeros(size(FRF,1),K));
previous = [];
for first = 1:groupWidth:K
    block = first:min(first+groupWidth-1,K);
    gram = mean(pagemtimes(whitened(:,:,block),'ctranspose', ...
        whitened(:,:,block),'none'),3);
    [vectors,values] = eig((gram+gram')/2,'vector');
    [~,best] = max(real(values));
    digital = txRoot\vectors(:,best);
    physical = FRF*digital;
    digital = digital/norm(physical);
    physical = FRF*digital;
    if ~isempty(previous)
        overlap = previous'*physical;
        rotation = exp(-1i*angle(overlap));
        digital = digital*rotation;
        physical = physical*rotation;
    end
    weights.FBB(:,block) = repmat(digital,1,numel(block));
    weights.Precoder(:,block) = repmat(physical,1,numel(block));
    previous = physical;
end
assert(max(abs(sum(abs(weights.Precoder).^2,1)-1)) < 1e-10);
assert(norm(weights.Receiver'*weights.Receiver-eye(size(WRF,2)),'fro') < 1e-9);
end
