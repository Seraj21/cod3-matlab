function r=v3Resources(cfg)
% Use all already transmitted references; no new pilot RE or pilot power.
r=muResources(cfg);
% Diagnostic alternative: disjoint pilot tones avoid frequency-CDM leakage.
% Preserve occupied RE, aggregate reference energy and the observation count.
r.OrthogonalDMRS=isfield(cfg,'OrthogonalDMRS') && cfg.OrthogonalDMRS;
if r.OrthogonalDMRS
 original=r.DMRS;r.DMRS(:)=0;offsets=[1 3 2 4];
 for user=1:cfg.Users
  times=r.DownlinkLayout{user}.Times;
  frequencies=offsets(user):4:r.K;
  r.DMRS(frequencies,times,user)=2*sign(original(frequencies,times,user));
  r.DownlinkLayout{user}=muPilotLayout(cfg.PRGWidth,r.L, ...
   offsets(user):4:cfg.PRGWidth,times,.25,r.DownlinkLayout{user}.Label);
 end
 assert(max(abs(sum(abs(r.DMRS).^2,3)-sum(abs(original).^2,3)),[],'all')<1e-10);
 assert(all(sum(abs(r.DMRS)>0,3)<=1,'all'));
 r.ReferenceGrid=r.DMRS+r.PTRS;
end
r.DMRSLegacyLayouts=r.DownlinkLayout;
for user=1:cfg.Users
 old=r.DownlinkLayout{user};p=old;
 p.Mask=old.Mask | abs(r.PTRS(1:cfg.PRGWidth,:,user))>0;
 p.Observed=find(p.Mask);p.Count=numel(p.Observed);p.Label=old.Label+"_P"+user;
 [f,t]=ind2sub([p.W p.L],p.Observed);coordinates=[double(f)/4 double(t)/3];
 tri=delaunayTriangulation(coordinates);
 [fq,tq]=ndgrid(1:p.W,1:p.L);query=[double(fq(:))/4 double(tq(:))/3];
 [which,bary]=pointLocation(tri,query);inside=find(~isnan(which));
 columns=tri.ConnectivityList(which(inside),:);rows=repmat(inside,1,3);values=bary(inside,:);
 operator=sparse(rows(:),columns(:),values(:),p.W*p.L,p.Count);
 outside=find(isnan(which));nearest=nearestNeighbor(tri,query(outside,:));
 operator=operator+sparse(outside,nearest,ones(numel(outside),1),p.W*p.L,p.Count);
 p.Linear=full(operator.');
 assert(p.Count==22 && max(abs(p.Linear(:,p.Observed)-eye(p.Count)),[],'all')<1e-10);
 assert(abs(p.NoiseGain-.25)<1e-12);p.NoiseGain=.25;
 % Isolated PTRS power and despread DMRS have the same observation-noise gain.
 values=r.PTRS(1:cfg.PRGWidth,:,user);values=values(values~=0);
 assert(all(abs(abs(values).^2-4)<1e-10));
 r.DownlinkLayout{user}=p;
end
r.PilotMode="DMRS+PTRS";
end
