function layouts=v3LSLayouts(r,mode)
% All modes receive the same 22 available observations.
mode=string(mode);layouts=r.DownlinkLayout;
if mode=="tensor plus PTRS innovation"
 layouts=v3ResidualLSLayouts(r);
elseif mode=="tensor DMRS"
 for user=1:4
  p=layouts{user};old=r.DMRSLegacyLayouts{user};
  [present,rows]=ismember(old.Observed,p.Observed);assert(all(present));
  p.Linear=zeros(p.Count,p.W*p.L);p.Linear(rows,:)=old.Linear;layouts{user}=p;
 end
else,assert(mode=="triangular full pilots");end
end
