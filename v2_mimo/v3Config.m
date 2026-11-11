function cfg=v3Config(antennas,numRB,profile)
if nargin<2,numRB=66;end
if nargin<3,profile="mobile";end
cfg=v2Config(antennas,numRB);cfg.Version=6;cfg.Rate=3/4;
profile=string(profile);assert(ismember(profile,["mobile" "stressed"]));
cfg.OperatingProfile=char(profile);
if profile=="mobile"
 cfg.DelaySpread=300e-9;cfg.Speed=30;cfg.SoundingSNR=30;
end
cfg.PilotMode='DMRS+PTRS';cfg.PZFMode='robust SLNR, quantized desired-phase anchor';
cfg.ReferenceObservationsPerLocalChannel=22;
end
