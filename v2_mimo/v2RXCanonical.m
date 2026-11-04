function [samples,truth,layouts,order]=v2RXCanonical(input,r,user,clean)
if nargin<4,clean=false;end
rows=(user-1)*4+(1:4);order=[user setdiff(1:4,user,'stable')];
source=input.Samples;if clean,source=input.CleanSamples;end
Q=fft(eye(4))/2;
samples=toBeams(source(:,:,rows,order),Q);
truth=[];if ~isempty(input.Truth),truth=toBeams(input.Truth(:,:,rows,order),Q);end
layouts=r.DownlinkLayout(order);
for u=1:4
 p=layouts{u};role="Other";if u==1,role="Own";end
 p.Label=role+"_"+p.Label;layouts{u}=p;
end
end
function y=toBeams(y,Q)
[K,L,R,U]=size(y);
y=permute(reshape(reshape(permute(y,[1 2 4 3]),[],R)*conj(Q),K,L,U,R),[1 2 4 3]);
end
