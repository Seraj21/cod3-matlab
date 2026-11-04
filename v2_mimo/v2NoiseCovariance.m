function R=v2NoiseCovariance(observed,h,variance,noise,r)
% Known DM-RS/PTRS reveal spatially correlated residual interference.
% These samples contain no unknown payload symbols and need no true channel.
[K,L,branches,users]=size(h);
prediction=sum(h.*reshape(r.ReferenceGrid,K,L,1,users),4);
mask=any(abs(r.ReferenceGrid)>0,3);
mask(:,1:6)=false; % The first DL pilot follows silence and understates payload ISI.
residual=(observed-prediction).*mask;
e=reshape(permute(residual,[3 1 2]),branches,1,K,L);
outer=e.*permute(conj(e),[2 1 3 4]);
outer=movsum(movsum(outer,[36 35],3),[2 2],4);
count=movsum(movsum(double(mask),[36 35],1),[2 2],2);
R=outer./reshape(max(count,1),1,1,K,L);
R=.85*R;
minimum=sum(variance,4);
for branch=1:branches
 raw=real(reshape(outer(branch,branch,:,:),K,L))./max(count,1);
 floorValue=noise(branch)+minimum(:,:,branch);
 R(branch,branch,:,:)=reshape(max(raw,floorValue),1,1,K,L);
end
end
