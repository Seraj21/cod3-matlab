function front = muFrontEnd(antennas,rfChains,verticalCorrelation,horizontalCorrelation)
% Partially connected UPA, one vertical subarray per RF chain.
assert(mod(antennas,rfChains)==0);
rows = antennas/rfChains;
F = kron(eye(rfChains),ones(rows,1)/sqrt(rows));
vertical = toeplitz(verticalCorrelation.^(0:rows-1));
horizontal = toeplitz(horizontalCorrelation.^(0:rfChains-1));
physical = kron(horizontal,vertical);
effective = F'*physical*F;
gain = sqrt(real(diag(effective)));
normalized = effective./(gain*gain.');
front = struct('FRF',F,'Correlation',normalized,'ChannelGain',gain, ...
    'Rows',rows,'Columns',rfChains,'PhysicalCorrelation',physical);
assert(norm(F'*F-eye(rfChains),'fro')<1e-12);
assert(max(abs(diag(normalized)-1))<1e-12);
end
