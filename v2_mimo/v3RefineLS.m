function model=v3RefineLS(model,rxRoot,folder)
% Select a fair LS comparator without changing learned LMMSE/CNN features.
% Channel calibration: original 80 held-out channels/precoder, seeds 18M.
% Operator selection: the eight predeclared payload-validation frames, seeds 19M.
modes=["tensor DMRS" "tensor plus PTRS innovation" "triangular full pilots"];
schemes=["MR" "ZF" "PZF"];banks=cell(3,3);calibration=cell(0,5);
cfg=v3Config(32,12);r=v3Resources(cfg);
for p=1:3
 scheme=schemes(p);rx=model.Receivers.(scheme);
 files=dir(fullfile(rxRoot,'v3-receiver-'+scheme,'rx_conditions.csv'));assert(numel(files)==1);
 conditions=readtable(fullfile(files.folder,files.name));conditions=conditions(~conditions.Training,:);
 assert(height(conditions)==80 && all(conditions.Seed>18000000 & conditions.Seed<19000000));
 keys=string(fieldnames(rx.PriorsRX));sums=zeros(numel(keys),3,2);
 for c=1:3,banks{p,c}=v3LSMoments(rx.PriorsRX,v3LSLayouts(r,modes(c)));end
 for index=1:height(conditions)
  row=conditions(index,:);cfg=v2ConfigFromRow(row,12);cfg.Rate=.125;scene=v2Sounding(cfg,r,row.Seed);
  if row.CSITSource==4,csit=scene.ULTruth;variance=zeros(size(csit));
  else
   tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
   csit=tx.Wiener;variance=tx.VarianceWiener;
   if row.CSITSource==1,csit=tx.LS;variance=tx.VarianceLS;end
  end
  weights=v3Weights(csit,variance,scheme,scene,r);
  [grid,~]=muPayload(cfg,r,muSeed(scene.Seed,70));packet=v2Waveform(scene,r,grid,weights,true);
  for snr=[row.DataSNR1 row.DataSNR2]
   input=v3Receive(packet,scene,r,snr,false,false);
   for c=1:3
    local=r;local.DownlinkLayout=v3LSLayouts(r,modes(c));
    for user=1:4
     [samples,truth,layouts]=v2RXCanonical(input,local,user);
     state=v2CEBase(samples,input.Noise((user-1)*4+(1:4)),layouts,banks{p,c});
     for u=1:4
      for branch=1:4
       k=find(keys==layouts{u}.Label+"_R"+branch);assert(isscalar(k));
       e=truth(:,6:12,branch,u)-state.LS(:,6:12,branch,u);
       sums(k,c,1)=sums(k,c,1)+sum(abs(e).^2,'all');
       sums(k,c,2)=sums(k,c,2)+sum(state.RawVarianceLS(:,6:12,branch,u),'all');
      end
     end
    end
   end
  end
  if mod(index,16)==0,fprintf('LS covariance calibration %s %d/80\n',scheme,index);end
 end
 for c=1:3
  prior=banks{p,c};
  for k=1:numel(keys)
   factor=max(1e-4,sums(k,c,1)/max(sums(k,c,2),eps));prior.(keys(k)).VarianceFactor.LS=factor;
   calibration(end+1,:)={scheme,modes(c),keys(k),factor,height(conditions)}; %#ok<AGROW>
  end
  banks{p,c}=prior;
 end
end
writetable(cell2table(calibration,'VariableNames',{'Precoder','LSInterpolation','PriorKey', ...
 'VarianceFactor','IndependentValidationChannels'}),fullfile(folder,'v3_ls_covariance_calibration.csv'));
points=[0 4 8 12 16 20];totals=zeros(3,3,3);rows=cell(0,9);
for bs=[32 64]
 cfg=v3Config(bs);r=v3Resources(cfg);
 for frame=1:4
  scene=v2Sounding(cfg,r,19000000+bs*1000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
  tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
  for p=1:3
   scheme=schemes(p);weights=v3Weights(tx.LS,tx.VarianceLS,scheme,scene,r);
   packet=v2Waveform(scene,r,grid,weights,false);
   for snr=points
    originalInput=v3Receive(packet,scene,r,snr,false,false);
    for c=1:3
     input=originalInput;
     rx=model.Receivers.(scheme);rx.LSInterpolation=modes(c);rx.PriorsLS=banks{p,c};
     [h,v,observed]=v3EstimateRX(input,rx,r,"LS");input.Observed=observed;
     [e,f]=v2Decode(input,h,v,payload,cfg,r);correct=nnz(e==0 & ~f);bits=4*r.PayloadBits;
     totals(c,p,:)=totals(c,p,:)+reshape([sum(e) bits correct],1,1,3);
     rows(end+1,:)={bs,frame,scene.Seed,scheme,modes(c),snr,sum(e),bits,correct}; %#ok<AGROW>
    end
   end
  end
  fprintf('LS independent payload validation BS%d frame%d/4 complete\n',bs,frame);
 end
end
for p=1:3
 score=[totals(:,p,1)./totals(:,p,2) -totals(:,p,3) (1:3).'];[~,order]=sortrows(score,[1 2 3]);c=order(1);
 scheme=schemes(p);model.LSInterpolation.(scheme)=char(modes(c));
 model.LSReferencesUsed.(scheme)=22;if c==1,model.LSReferencesUsed.(scheme)=18;end
 model.Receivers.(scheme).LSInterpolation=modes(c);model.Receivers.(scheme).PriorsLS=banks{p,c};
 fprintf('Selected LS %s: %s on held-out payload BER\n',scheme,modes(c));
end
writetable(cell2table(rows,'VariableNames',{'BS','Frame','ChannelSeed','Precoder', ...
 'LSInterpolation','SNR_dB','PayloadBitErrors','PayloadBits','CorrectTransportBlocks'}),fullfile(folder,'v3_ls_payload_validation.csv'));
model.LSValidation=struct('FramesPerArray',4,'SNR',points,'Candidates',modes,'Counts',totals, ...
 'SeedBase',19000000,'SelectionRule',"minimum mean payload BER, then correct blocks, then earlier declared operator");
end
