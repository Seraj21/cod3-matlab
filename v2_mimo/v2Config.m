function cfg=v2Config(antennas,numRB)
if nargin<2,numRB=66;end
cfg=muConfig(antennas,NumRB=numRB);
cfg.Version=5;
cfg.BeamWidth=1;
cfg.CPFraction=.75;
cfg.AlignPhase=true;
cfg.DecoderDual=true;
end
