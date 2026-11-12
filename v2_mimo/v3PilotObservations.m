function samples=v3PilotObservations(observed,r)
branches=size(observed,3);
if isfield(r,'OrthogonalDMRS') && r.OrthogonalDMRS
 samples=complex(zeros(r.K,r.L,branches,4));
else
 samples=muPilotObservations(observed,r,"DL");
end
flat=reshape(observed,r.K*r.L,branches);
for user=1:4
 reference=r.PTRS(:,:,user);
 if isfield(r,'OrthogonalDMRS') && r.OrthogonalDMRS,reference=reference+r.DMRS(:,:,user);end
 indices=find(reference);
 temp=reshape(samples(:,:,:,user),r.K*r.L,branches);
 temp(indices,:)=flat(indices,:)./reference(indices);
 samples(:,:,:,user)=reshape(temp,r.K,r.L,branches);
end
end
