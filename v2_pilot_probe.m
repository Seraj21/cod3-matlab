function v2_pilot_probe
% Check whether beam updates invalidate the effective-channel pilot model.
addpath('mu_mimo','v2_mimo');load('v2_tx_training/v2_tx.mat','model');
cfg=v2Config(64);r=muResources(cfg);rows=cell(0,10);
for frame=1:2
 scene=v2Sounding(cfg,r,6450000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 practical=v2EstimateTX(scene.ULSamples,scene.ULNoise,model,r,"LMMSE");
 for source=["Perfect" "LMMSE"]
  original=scene.ULTruth;if source=="LMMSE",original=practical;end
  for width=[1 4 12 24]
   scene.Config.BeamWidth=width;
   for heldTime=[0 6 9]
    csit=original;
    if heldTime>0,csit=repmat(original(:,heldTime,:,:),1,r.L,1,1);end
    weights=v2Weights(csit,"ZF",scene,r);packet=v2Waveform(scene,r,grid,weights,true);
    input=v2Receive(packet,scene,r,20,false,false);
    truth=beamChannels(input.Truth);custom=beamChannels(muLS(input.Samples,r.DownlinkLayout));
    native=beamChannels(nrChannelEstimate(r.Carrier,input.Observed,r.DMRS, ...
        'CDMLengths',[2 1],'AveragingWindow',[1 1]));
    averaged=beamChannels(nrChannelEstimate(r.Carrier,input.Observed,r.DMRS,'CDMLengths',[2 1]));
    observed=v2BeamObserved(input.Observed,r);input.Observed=observed;
    estimates={truth,custom,native,averaged};names=["Perfect" "Custom LS" "NR LS" "NR averaged LS"];
    td=reshape(truth,r.K*r.L,16,4);td=td(r.DataIndices,:,:);
    for method=1:4
     h=estimates{method};d=reshape(h,r.K*r.L,16,4);d=d(r.DataIndices,:,:);
     nmse=sum(abs(d-td).^2,'all')/sum(abs(td).^2,'all');
     [e,f]=v2Decode(input,h,zeros(size(h)),payload,scene.Config,r);
     correct=nnz(e==0 & ~f);
     rows(end+1,:)={frame,source,width,heldTime,names(method),sum(e),numel(payload), ...
      correct,correct*r.PayloadBits/r.FrameDuration/1e6,nmse}; %#ok<AGROW>
    end
   end
  end
 end
 fprintf('Completed pilot/precoder physical control %d/2\n',frame);
end
T=cell2table(rows,'VariableNames',{'Frame','CSIT','BeamWidth','HeldCSITSymbol','Receiver', ...
 'Errors','Bits','CorrectBlocks','Goodput_Mbps','DataChannelNMSE'});
writetable(T,'v2_pilot_probe.csv');
fprintf('V2_PILOT_PROBE_BEGIN\n%s\nV2_PILOT_PROBE_END\n',jsonencode(table2struct(T)));
end
function output=beamChannels(h)
[K,L,R,U]=size(h);output=complex(zeros(K,L,R,U));Q=fft(eye(4))/2;
for user=1:4
 rows=(user-1)*4+(1:4);x=h(:,:,rows,:);
 output(:,:,rows,:)=permute(reshape(reshape(permute(x,[1 2 4 3]),[],4)*conj(Q),K,L,U,4),[1 2 4 3]);
end
end
