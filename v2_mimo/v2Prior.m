function output=v2Prior(action,varargin)
% Empirical affine LMMSE moments; optional separate UE receive-beam statistics.
switch action
 case "init"
  layouts=varargin{1};branches=varargin{2};output=struct;
  for u=1:numel(layouts)
   p=layouts{u};keys=string(p.Label);
   if branches>0,keys=p.Label+"_R"+(1:branches);end
   for key=keys
    if ~isfield(output,key)
     output.(key)=struct('XX',complex(zeros(p.Count)), ...
      'XY',complex(zeros(p.Count,p.W*p.L)),'YY',zeros(1,p.W*p.L), ...
      'SX',complex(zeros(1,p.Count)),'SY',complex(zeros(1,p.W*p.L)), ...
      'Count',0,'Layout',p);
    end
   end
  end
 case "accumulate"
  output=varargin{1};samples=varargin{2};truth=varargin{3};layouts=varargin{4};
  [K,L,R,U]=size(truth);W=layouts{1}.W;G=K/W;
  for u=1:U
   p=layouts{u};separate=~isfield(output,p.Label);
   if separate,groups=num2cell(1:R);else,groups={1:R};end
   for group=1:numel(groups)
    branches=groups{group};key=p.Label;
    if separate,key=key+"_R"+branches;end
    x=patchRows(samples(:,:,branches,u),W);x=x(:,p.Observed);
    y=patchRows(truth(:,:,branches,u),W);
    scale=sqrt(max(mean(abs(x).^2,2),1e-5));x=x./scale;y=y./scale;
    s=output.(key);s.XX=s.XX+x'*x;s.XY=s.XY+x'*y;
    s.YY=s.YY+sum(abs(y).^2,1);s.SX=s.SX+sum(x,1);s.SY=s.SY+sum(y,1);
    s.Count=s.Count+G*numel(branches);output.(key)=s;
   end
  end
 case "finalize"
  stats=varargin{1};output=struct;
  for key=string(fieldnames(stats)).'
   s=stats.(key);p=s.Layout;assert(s.Count>0);
   mx=s.SX/s.Count;my=s.SY/s.Count;
   R=s.XX/s.Count-mx'*mx;R=(R+R')/2;
   C=s.XY/s.Count-mx'*my;Y=max(0,s.YY/s.Count-abs(my).^2);
   R=.999*R+.001*max(real(trace(R))/p.Count,1e-5)*eye(p.Count);
   [V,D]=eig(R,'vector');D=max(real(D),1e-8);
   bias=Y+real(sum(conj(p.Linear).*(R*p.Linear),1)) ...
       -2*real(sum(conj(p.Linear).*C,1))+abs(my-mx*p.Linear).^2;
   output.(key)=struct('V',V,'D',D.','Cross',V'*C,'YY',Y,'MeanX',mx,'MeanY',my, ...
    'LSBias',max(0,bias),'LSNoise',sum(abs(p.Linear).^2,1), ...
    'VarianceFactor',struct('LS',1,'LMMSE',1,'CNN',1));
  end
 otherwise,error('Unknown prior operation');
end
end
function rows=patchRows(h,W)
[K,L,R]=size(h);G=K/W;
rows=reshape(permute(reshape(double(h),W,G,L,R),[1 3 2 4]),W*L,[]).';
end
