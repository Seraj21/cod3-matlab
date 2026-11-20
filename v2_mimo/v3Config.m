function cfg=v3Config(antennas,numRB,profile,matchedPilots,densePilots)
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
cfg.DMRSTimes=[6 9 12];cfg.ReferenceObservationsPerLocalChannel=22;
cfg.DensePilots=nargin>=5 && densePilots;
if cfg.DensePilots
 assert(matchedPilots);
 cfg.UplinkSymbols=1:2;cfg.GuardSymbols=3;cfg.DownlinkSymbols=4:12;
 % Five experimental downlink sounding symbols bracket the same four data symbols.
 cfg.DMRSTimes=4:2:12;cfg.ReferenceObservationsPerLocalChannel=34;
end
end
