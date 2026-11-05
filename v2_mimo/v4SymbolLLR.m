function llr=v4SymbolLLR(symbols,variance,desiredError,M)
% Marginal likelihood for z = s + e*s + n with complex Gaussian errors.
% The supplied variance is for an average-power QAM symbol.
symbols=double(symbols(:));variance=double(variance(:));
floorVariance=1e-7;
variance=max(floorVariance,variance);
desiredError=max(0,min(double(desiredError(:)),variance-floorVariance));
assert(numel(symbols)==numel(variance) && numel(symbols)==numel(desiredError));
persistent order constellation energy bit0
if isempty(order) || order~=M
 order=M;Qm=log2(M);
 indices=(0:M-1).';
 constellation=qammod(indices,M,'UnitAveragePower',true).';
 energy=abs(constellation).^2;
 bit0=zeros(M,Qm);
 for k=1:Qm,bit0(:,k)=bitget(uint16(indices),Qm-k+1)==0;end
end
Qm=size(bit0,2);
llr=zeros(Qm,numel(symbols));
for start=1:128:numel(symbols)
 ids=start:min(start+127,numel(symbols));
 conditionalVariance=variance(ids)+desiredError(ids).*(energy-1);
 cost=abs(symbols(ids)-constellation).^2./conditionalVariance+log(conditionalVariance);
 probability=exp(min(cost,[],2)-cost);
 p0=probability*bit0;p1=probability*(1-bit0);
 llr(:,ids)=(log(max(p0,realmin))-log(max(p1,realmin))).';
end
llr=llr(:);
end
