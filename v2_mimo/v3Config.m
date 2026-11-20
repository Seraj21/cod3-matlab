function cfg=v3Config(antennas,numRB,profile,matchedPilots)
if nargin<2,numRB=66;end
if nargin<3,profile="stressed";end
cfg=v2Config(antennas,numRB);cfg.Version=6;cfg.Rate=3/4;
profile=string(profile);assert(ismember(profile,["mobile" "stressed"]));
cfg.OperatingProfile=char(profile);
cfg.PrecoderTimeMode='per-symbol';
if profile=="mobile"
 cfg.DelaySpread=300e-9;cfg.Speed=30;cfg.SoundingSNR=30;
end
if nargin>=4 && matchedPilots
 assert(profile=="mobile" && numRB==264);
 cfg.BeamWidth=24;cfg.PrecoderTimeMode='slot';
 cfg.OrthogonalDMRS=true;cfg.StaggeredDMRS=true;cfg.MatchedPilotMask=true;
end
cfg.PilotMode='DMRS+PTRS';cfg.PZFMode='robust SLNR, quantized desired-phase anchor';
cfg.ReferenceObservationsPerLocalChannel=22;
end
