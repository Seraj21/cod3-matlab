function cfg = muConfig(antennas,options)
arguments
    antennas (1,1) double {mustBeMember(antennas,[32 64])} = 32
    options.NumRB (1,1) double = 66
    options.Users (1,1) double = 4
    options.UERx (1,1) double = 4
    options.RFChains (1,1) double = 8
    options.DelaySpread (1,1) double = 1e-6
    options.Speed (1,1) double = 120
    options.VerticalCorrelation (1,1) double = .7
    options.HorizontalCorrelation (1,1) double = .3
    options.UECorrelation (1,1) double = .25
    options.SoundingSNR (1,1) double = 15
    options.Quiet (1,1) logical = false
end
cfg = options;
cfg.Version = 4;
cfg.Antennas = antennas;
cfg.M = 512;
cfg.Qm = 9;
cfg.Rate = .5;
cfg.CarrierFrequency = 28e9;
cfg.SubcarrierSpacing = 60;
cfg.PhaseBits = 4;
cfg.PRGWidth = 24;
cfg.UplinkSymbols = 1:4;
cfg.GuardSymbols = 5;
cfg.DownlinkSymbols = 6:12;
cfg.PhaseNoiseLevel = [-85 -100 -115];
cfg.PhaseNoiseOffset = [1e4 1e5 1e6];
cfg.InputBackoff = 10;
cfg.AMPMDegrees = 3;
cfg.FrequencyOffset = 2800;
if cfg.Quiet
    cfg.PhaseNoiseLevel = [-300 -300 -300];
    cfg.InputBackoff = 80;
    cfg.AMPMDegrees = 0;
    cfg.FrequencyOffset = 0;
end
assert(cfg.Users<=cfg.RFChains && cfg.Users<=4 && mod(antennas,cfg.RFChains)==0);
assert(mod(12*cfg.NumRB,cfg.PRGWidth)==0);
end
