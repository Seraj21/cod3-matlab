function item=v2CropRecord(samples,truth,noise,frame,seed,r)
bands=sort(1+mod(frame-1+[0 floor(r.G/2)],r.G));
indices=reshape(((bands(:)-1)*24+(1:24)).',[],1);
item=struct('Samples',single(samples(indices,:,:,:)),'Truth',single(truth(indices,:,:,:)), ...
 'Noise',noise,'Seed',seed);
end
