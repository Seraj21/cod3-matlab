function model=v3SelectPayload(model,folder)
% Three predeclared global RX multipliers on eight held-out channel frames.
candidates=[0 .5 1];points=[0 4 8 12 16 20];names=["MR" "ZF" "PZF"];
totals=zeros(3,3,3);raw=cell(0,8);
for bs=[32 64]
 cfg=v3Config(bs);r=v3Resources(cfg);
 for frame=1:4
  scene=v2Sounding(cfg,r,19000000+bs*1000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
  state=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
  for p=1:3
   scheme=names(p);rx=model.Receivers.(scheme);
   weights=v3Weights(state.Wiener,state.VarianceWiener,scheme,scene,r);packet=v2Waveform(scene,r,grid,weights,false);
   for snr=points
    input=v3Receive(packet,scene,r,snr,false,false);
    [lm,lmv,observed]=v2EstimateRX(input,rx,r,"LMMSE");[cnn,cnnv]=v2EstimateRX(input,rx,r,"CNN");input.Observed=observed;
    for a=1:3
     alpha=candidates(a);h=lm+alpha*(cnn-lm);v=lmv+alpha*(cnnv-lmv);
     [e,f]=v2Decode(input,h,v,payload,cfg,r);correct=nnz(e==0 & ~f);bits=4*r.PayloadBits;
     totals(a,p,:)=totals(a,p,:)+reshape([sum(e) bits correct],1,1,3);
     raw(end+1,:)={bs,frame,scheme,alpha,snr,sum(e),bits,correct}; %#ok<AGROW>
    end
   end
  end
  fprintf('V3 independent payload validation BS%d frame%d/4 complete\n',bs,frame);
 end
end
for p=1:3
 score=[totals(:,p,1)./totals(:,p,2) -totals(:,p,3) candidates.'];[~,order]=sortrows(score,[1 2 3]);
 model.Selection.(names(p))=[0 candidates(order(1))];
end
writetable(cell2table(raw,'VariableNames',{'BS','Frame','Precoder','RXMultiplier','SNR_dB', ...
 'PayloadBitErrors','PayloadBits','CorrectTransportBlocks'}),fullfile(folder,'v3_payload_validation.csv'));
model.PayloadValidation=struct('FramesPerArray',4,'SNR',points,'Candidates',candidates, ...
 'Counts',totals,'SeedBase',19000000,'SelectionRule',"minimum mean payload BER, then correct blocks, then smaller correction");
end
