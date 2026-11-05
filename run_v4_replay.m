function run_v4_replay(antennas,folder)
% Check call-order independence and V3/V4 equivalence inside one MATLAB run.
addpath(pwd,'mu_mimo','v2_mimo');
cfg=v3Config(antennas);r=v3Resources(cfg);cfg.SampleRate=r.OFDM.SampleRate;
clear muPhaseNoise
for count=[r.Samples r.Samples+ceil(12e-6*cfg.SampleRate)]
 first=muPhaseNoise(count,cfg,581713);
 muPhaseNoise(count,cfg,581714);
 repeated=muPhaseNoise(count,cfg,581713);
 fprintf('Oscillator first vs repeat count %d: max absolute difference %.9g\n', ...
  count,max(abs(first-repeated)));
end
clear muPhaseNoise
first=v2Sounding(cfg,r,50000001);
repeated=v2Sounding(cfg,r,50000001);
fprintf('SRS pilot repeat difference %.9g; data noise repeat difference %.9g\n', ...
 max(abs(first.ULSamples-repeated.ULSamples),[],'all'), ...
 max(abs(first.DataNoise-repeated.DataNoise),[],'all'));
clear first repeated muPhaseNoise
points=[0 10 20];
v3Folder=fullfile(folder,'v3');v4Folder=fullfile(folder,'v4');
run_v3_mimo(antennas,1,v3Folder,points,50000000);
clear muPhaseNoise
run_v4_pzf(antennas,1,v4Folder,points,50000000);
v3=readtable(fullfile(v3Folder,'mu_slot_errors.csv'),'TextType','string');
v4=readtable(fullfile(v4Folder,'pzf_slots.csv'),'TextType','string');
v3=v3(v3.Precoder=="PZF",:);
v3.Estimator(v3.Estimator=="CNN")="V3 CNN";
v3=sortrows(v3,{'SNR_dB','Estimator','User'});
v4=sortrows(v4,{'SNR_dB','Estimator','UE'});
assert(isequal(v3.SNR_dB,v4.SNR_dB) && isequal(v3.Estimator,v4.Estimator));
assert(isequal(v3.User,v4.UE));
errors=nnz(v3.PayloadBitErrors~=v4.Errors);
crc=nnz(v3.CRCFailed~=v4.CRCFailed);
beam=nnz(v3.SRSBeamAccepted~=v4.SRSBeamAccepted);
fprintf('Same-run V3 vs V4 BS%d: %d records; errors %d CRC %d beam %d mismatches\n', ...
 antennas,height(v3),errors,crc,beam);
assert(errors==0 && crc==0 && beam==0,'PZF baseline replay changed inside the same MATLAB environment');
end
