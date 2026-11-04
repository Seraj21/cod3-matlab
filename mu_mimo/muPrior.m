function varargout = muPrior(action,varargin)
switch action
    case "init"
        layouts = varargin{1};
        stats = struct;
        for u = 1:numel(layouts)
            p = layouts{u};
            key = char(p.Label);
            stats.(key) = struct('XX',complex(zeros(p.Count)), ...
                'XY',complex(zeros(p.Count,p.W*p.L)), ...
                'YY',zeros(1,p.W*p.L),'Count',0,'Layout',p);
        end
        varargout{1} = stats;
    case "accumulate"
        stats = varargin{1};
        observations = varargin{2};
        truth = varargin{3};
        layouts = varargin{4};
        W = layouts{1}.W;
        L = layouts{1}.L;
        [K,~,R,U] = size(truth);
        G = K/W;
        for u = 1:U
            p = layouts{u};
            x = muPatchRows(observations(:,:,:,u),W);
            y = muPatchRows(truth(:,:,:,u),W);
            x = x(:,p.Observed);
            scale = sqrt(max(mean(abs(x).^2,2),1e-5));
            x = x./scale;
            y = y./scale;
            key = char(p.Label);
            s = stats.(key);
            s.XX = s.XX+x'*x;
            s.XY = s.XY+x'*y;
            s.YY = s.YY+sum(abs(y).^2,1);
            s.Count = s.Count+G*R;
            stats.(key) = s;
        end
        varargout{1} = stats;
    case "finalize"
        stats = varargin{1};
        models = struct;
        for key = string(fieldnames(stats)).'
            s = stats.(key);
            assert(s.Count>0);
            p = s.Layout;
            R = (s.XX+s.XX')/(2*s.Count);
            R = .999*R+.001*max(real(trace(R))/p.Count,1e-5)*eye(p.Count);
            C = s.XY/s.Count;
            Y = s.YY/s.Count;
            [V,D] = eig((R+R')/2,'vector');
            D = max(real(D),1e-8);
            cross = V'*C;
            bias = Y+real(sum(conj(p.Linear).*(R*p.Linear),1)) ...
                -2*real(sum(conj(p.Linear).*C,1));
            models.(key) = struct('V',V,'D',D.','Cross',cross,'YY',Y, ...
                'LSBias',max(0,bias),'LSNoise',sum(abs(p.Linear).^2,1), ...
                'VarianceFactor',struct('LS',1,'LMMSE',1,'CNN',1));
        end
        varargout{1} = models;
    otherwise
        error('Unknown prior action');
end
end
function rows = muPatchRows(h,W)
[K,L,R] = size(h);
G = K/W;
rows = reshape(permute(reshape(double(h),W,G,L,R),[1 3 2 4]),W*L,[]).';
end
