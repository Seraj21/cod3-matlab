function loss=v2RXLoss(predictions,targets)
C=size(predictions,3)/3;
error=predictions(:,:,1:2*C,:)-targets(:,:,1:2*C,:);
error2=error(:,:,1:C,:).^2+error(:,:,C+(1:C),:).^2;
logvar=max(-10,min(5,targets(:,:,2*C+(1:C),:)+predictions(:,:,2*C+(1:C),:)));
weights=reshape(single([0 0 0 0 0 1 1 1 1 1 1 1]),1,12,1,1);
mse=mean(error2.*weights,'all')/mean(weights,'all');
nll=mean((error2.*exp(-logvar)+logvar).*weights,'all')/mean(weights,'all');
loss=mse+.015*nll;
end
