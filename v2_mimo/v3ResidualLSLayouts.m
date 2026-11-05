function layouts=v3ResidualLSLayouts(r)
% Preserve tensor-product DMRS interpolation; interpolate only PTRS innovation.
% This fixes an avoidable global change to LS away from the extra references.
layouts=r.DownlinkLayout;
for user=1:4
 p=layouts{user};old=r.DMRSLegacyLayouts{user};
 [present,legacyRows]=ismember(old.Observed,p.Observed);assert(all(present));
 additional=setdiff(1:p.Count,legacyRows,'stable');assert(numel(additional)==4);
 base=zeros(p.Count,p.W*p.L);base(legacyRows,:)=old.Linear;
 selector=zeros(p.Count,numel(additional));
 selector(sub2ind(size(selector),additional(:),(1:numel(additional)).'))=1;
 innovation=selector-base(:,p.Observed(additional));
 p.Linear=base+innovation*p.Linear(additional,:);
 assert(max(abs(p.Linear(:,p.Observed)-eye(p.Count)),[],'all')<1e-10);
 layouts{user}=p;
end
end
