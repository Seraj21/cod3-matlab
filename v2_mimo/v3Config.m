function cfg=v3Config(antennas,numRB)
if nargin<2,numRB=66;end
cfg=v2Config(antennas,numRB);cfg.Version=6;cfg.Rate=.125;
cfg.PilotMode='DMRS+PTRS';cfg.PZFMode='robust SLNR, quantized desired-phase anchor';
cfg.ReferenceObservationsPerLocalChannel=22;
end
