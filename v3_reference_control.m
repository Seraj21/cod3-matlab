function v3_reference_control
% Diagnose all-reference interpolation independently of the learned receiver.
% Truth is used only to score this diagnostic; no truth enters either LS path.
addpath(pwd,'mu_mimo','v2_mimo');load('v2_frozen/v2_model.mat','model');
rows=cell(0,13);variants=["MR" "ZF" "PZF baseline" "PZF V3"];
for bs=[32 64]
 cfg=v3Config(bs);r=v3Resources(cfg);
 for frame=1:4
  scene=v2Sounding(cfg,r,6900000+bs*1000+frame);
  [grid,~]=muPayload(cfg,r,muSeed(scene.Seed,70));
  tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
  for k=1:numel(variants)
   scheme=variants(k);if startsWith(scheme,"PZF"),scheme="PZF";end
   if k==4,[weights,d]=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
   else,[weights,d]=v2Weights(tx.Wiener,scheme,scene,r);end
   packet=v2Waveform(scene,r,grid,weights,true);
   for snr=[0 8 16 20]
    input=v3Receive(packet,scene,r,snr,false,true);
    diagonal=sum(input.Truth.*reshape(r.ReferenceGrid,r.K,r.L,1,4),4);
    observations={diagonal,input.CleanObserved,input.Observed};
    labels=["Noiseless diagonal pilots" "Noiseless actual waveform" "Noisy actual waveform"];
    for q=1:3
     old=muLS(muPilotObservations(observations{q},r,"DL"),r.DMRSLegacyLayouts);
     full=muLS(v3PilotObservations(observations{q},r),r.DownlinkLayout);
     [oldBeam,oldAll]=ownNMSE(old,input.Truth,r);
     [fullBeam,fullAll]=ownNMSE(full,input.Truth,r);
     rows(end+1,:)={bs,frame,scene.Seed,variants(k),snr,labels(q), ...
      oldBeam,fullBeam,oldAll,fullAll,18,22,d.MaximumPowerError}; %#ok<AGROW>
    end
   end
  end
  T=cell2table(rows,'VariableNames',{'BSAntennas','Frame','ChannelSeed','Variant', ...
   'SNR_dB','Observation','DMRS_OwnBeamNMSE','FullPilot_OwnBeamNMSE', ...
   'DMRS_AllRxNMSE','FullPilot_AllRxNMSE','DMRSCount','FullPilotCount','PowerError'});
  writetable(T,'v3_reference_control.csv');
  fprintf('V3 independent reference control BS%d frame%d/4 complete\n',bs,frame);
 end
end
end
function [beam,allRx]=ownNMSE(h,truth,r)
a=0;b=0;c=0;d=0;Q=fft(eye(4))/2;
for u=1:4
 rows=(u-1)*4+(1:4);x=reshape(h(:,:,rows,u),[],4)*conj(Q);
 y=reshape(truth(:,:,rows,u),[],4)*conj(Q);ix=r.DataIndices;
 a=a+sum(abs(x(ix,1)-y(ix,1)).^2);b=b+sum(abs(y(ix,1)).^2);
 c=c+sum(abs(x(ix,:)-y(ix,:)).^2,'all');d=d+sum(abs(y(ix,:)).^2,'all');
end
beam=a/max(b,eps);allRx=c/max(d,eps);
end
