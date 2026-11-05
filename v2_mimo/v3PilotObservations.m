function samples=v3PilotObservations(observed,r)
samples=muPilotObservations(observed,r,"DL");branches=size(observed,3);
flat=reshape(observed,r.K*r.L,branches);
for user=1:4
 reference=r.PTRS(:,:,user);indices=find(reference);
 temp=reshape(samples(:,:,:,user),r.K*r.L,branches);
 temp(indices,:)=flat(indices,:)./reference(indices);
 samples(:,:,:,user)=reshape(temp,r.K,r.L,branches);
end
end
