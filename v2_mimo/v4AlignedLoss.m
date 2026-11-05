function loss=v4AlignedLoss(predictions,targets)
% Evaluate the complex error after the LMMSE receive projection.
% Packed channels are ordered tone, receive branch, then transmit stream.
C=size(predictions,3)/3;
loss=v2RXLoss(predictions,targets(:,:,1:3*C,:));
er=predictions(:,:,1:C,:)-targets(:,:,1:C,:);
ei=predictions(:,:,C+(1:C),:)-targets(:,:,C+(1:C),:);
cr=targets(:,:,3*C+(1:C),:);ci=targets(:,:,4*C+(1:C),:);
realPart=er.*cr-ei.*ci;imagPart=er.*ci+ei.*cr;
idx=reshape((1:4).'+16*(0:3),1,[]);
pr=realPart(:,:,idx,:)+realPart(:,:,idx+4,:)+realPart(:,:,idx+8,:)+realPart(:,:,idx+12,:);
pi=imagPart(:,:,idx,:)+imagPart(:,:,idx+4,:)+imagPart(:,:,idx+8,:)+imagPart(:,:,idx+12,:);
weights=reshape(single([0 0 0 0 0 1 1 1 1 1 1 1]),1,12,1,1).* ...
 reshape(single([2*ones(1,4) ones(1,12)]),1,1,16,1);
loss=loss+.65*mean((pr.^2+pi.^2).*weights,'all')/mean(weights,'all');
end
