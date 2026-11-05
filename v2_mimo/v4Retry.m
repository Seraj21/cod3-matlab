function [decoded,failed,iterations]=v4Retry(primary,failed,cfg,r,mode)
% A retry can replace a block only after CRC passes. No true bits are used.
decoded=primary.Decoded;
iterations=zeros(1,cfg.Users);
for user=find(failed)
 if any(mode==["NMS50" "LBP50"])
  llr=primary.LLR(:,user);
 else
  desiredError=primary.ChannelError(:,user);
  if any(mode==["APP20" "APP50"]),desiredError=zeros(size(desiredError));end
  demapped=v4SymbolLLR(primary.Symbols(:,user),primary.Variance(:,user),desiredError,cfg.M);
  llr=zeros(r.CodedBits,1);llr(r.BitOrder)=max(-50,min(50,demapped));
 end
 if any(mode==["LBP50" "APP50" "CSI-APP50"])
  [bits,crcFailed,n]=r.Codec.decode(llr,50,'Layered belief propagation');
 elseif mode=="NMS50"
  [bits,crcFailed,n]=r.Codec.decode(llr,50);
 else
  [bits,crcFailed,n]=r.Codec.decode(llr);
 end
 iterations(user)=sum(n(:));
 if ~crcFailed
  decoded(:,user)=bits;
  failed(user)=false;
 end
end
end
