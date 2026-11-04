function v2_control
addpath('mu_mimo','v2_mimo');rows=cell(0,9);
for antennas=[32 64]
 cfg=v2Config(antennas);r=muResources(cfg);
 for index=1:4
  scene=muSounding(cfg,r,6200000+index);
  [grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
  for scheme=["MR" "ZF" "PZF"]
   [weights,d]=v2Weights(scene.ULTruth,scheme,scene,r);
   packet=v2Waveform(scene,r,grid,weights,true);
   for snr=[10 20]
    input=v2Receive(packet,scene,r,snr,true,false);
    h=complex(zeros(size(input.Truth)));
    for u=1:4
     [~,canonical,~,order]=v2RXCanonical(input,r,u);
     h(:,:,(u-1)*4+(1:4),order)=canonical;
    end
    input.Observed=v2BeamObserved(input.Observed,r);
    [e,f,details]=v2Decode(input,h,zeros(size(h)),payload,scene.Config,r);
    success=nnz(e==0 & ~f);
    rows(end+1,:)={antennas,index,scheme,snr,sum(e),numel(payload),success, ...
       success*r.PayloadBits/r.FrameDuration/1e6,nnz(details.SRSBeamAccepted)};
   end
  end
 end
end
T=cell2table(rows,'VariableNames',{'BS','Frame','Precoder','SNR_dB','Errors','Bits', ...
 'CorrectBlocks','Goodput_Mbps','SRSBeamAccepted'});
writetable(T,'v2_control.csv');
fprintf('V2_CONTROL_BEGIN\n%s\nV2_CONTROL_END\n',jsonencode(table2struct(T)));
end
