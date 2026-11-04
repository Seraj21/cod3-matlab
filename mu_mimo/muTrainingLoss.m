function loss = muTrainingLoss(predictions,targets,link)
if link=="TX"
    weights = single([.2 .2 .2 .2 .5 1 1 1 1 1 1 1]);
else
    weights = single([0 0 0 0 0 1 1 1 1 1 1 1]);
end
error = predictions-targets;
loss = mean(error.^2.*reshape(weights,1,12,1,1),'all')/mean(weights);
difference = error(:,2:end,:,:)-error(:,1:end-1,:,:);
w = min(weights(1:end-1),weights(2:end));
loss = loss+.03*mean(difference.^2.*reshape(w,1,11,1,1),'all')/mean(w);
end
