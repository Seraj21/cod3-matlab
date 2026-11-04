function [weights,diagnostic] = muWeights(h,scheme,scene,r)
cfg = scene.Config;
weights = complex(zeros(cfg.RFChains,cfg.Users,r.K,r.L,'single'));
rows = zeros(r.G*numel(cfg.DownlinkSymbols),4);
previous = cell(r.G,cfg.Users);
row = 0;
for l = cfg.DownlinkSymbols
    for g = 1:r.G
        band = (g-1)*cfg.PRGWidth+(1:cfg.PRGWidth);
        H = reshape(mean(h(band,l,:,:),1),cfg.RFChains,cfg.Users).';
        [B,d] = muPrecoder(double(H),scheme,scene.Front,cfg.PhaseBits,2);
        for u = 1:cfg.Users
            prior = previous{g,u};
            if isempty(prior) && g>1,prior=previous{g-1,u};end
            if ~isempty(prior)
                phase = angle(prior'*B(:,u));
                if scheme=="PZF"
                    step = 2*pi/2^cfg.PhaseBits;
                    phase = step*round(phase/step);
                end
                B(:,u) = B(:,u)*exp(-1i*phase);
            end
            previous{g,u} = B(:,u);
        end
        weights(:,:,band,l) = repmat(single(B),1,1,numel(band));
        row = row+1;
        rows(row,:) = [d.TotalPower d.LeakageRatio d.ConditionNumber d.ConstantModulusError];
    end
end
diagnostic = struct('MaximumPowerError',max(abs(rows(:,1)-1)), ...
    'MeanEstimatedLeakage',mean(rows(:,2)),'MaximumCondition',max(rows(:,3)), ...
    'MaximumConstantModulusError',max(rows(:,4)));
end
