function [banks,WRF,directions] = hybridCodebook(rows,columns,phaseBits)
% Half-wavelength UPA. Rows run vertically, then columns horizontally.
% u and v are the horizontal and vertical direction cosines.
[vertical,horizontal] = ndgrid([-1 0],-3:3);
bins = [vertical(:) horizontal(:); 1 -1; 1 1];
directions = [2*bins(:,2)/columns 2*bins(:,1)/rows];
assert(all(sum(directions.^2,2)<=1+1e-12));
[m,n] = ndgrid(0:rows-1,0:columns-1);
phase = pi*(m(:)*directions(:,2).'+n(:)*directions(:,1).');
step = 2*pi/2^phaseBits;
codebook = exp(1i*step*round(phase/step))/sqrt(rows*columns);
banks = {codebook(:,1:8),codebook(:,9:16)};
WRF = codebook(:,[2 4 6 8 10 12 14 15]);
assert(norm(WRF'*WRF-eye(8),'fro')<1e-9);
for bank = 1:numel(banks)
    assert(norm(banks{bank}'*banks{bank}-eye(8),'fro')<1e-9);
end
end
