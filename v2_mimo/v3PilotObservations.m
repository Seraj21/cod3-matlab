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
 if isfield(r,'StaggeredDMRS') && r.StaggeredDMRS && ~r.MatchedPilotMask
  % Frozen CNN features expect a fixed comb. Regrid observed coefficients
  % within each beam group for these features; the posterior uses raw pilots.
  layout=r.DMRSLegacyLayouts{user};W=layout.W;
  for l=layout.Times.'
   actual=find(r.DMRS(1:W,l,user));
   values=reshape(samples(:,l,:,user),W,[]);
   q=min(actual(end),max(actual(1),layout.Freq));
   values(layout.Freq,:)=interp1(actual,values(actual,:),q,'linear');
   samples(:,l,:,user)=reshape(values,r.K,1,branches);
  end
 end
end
end
