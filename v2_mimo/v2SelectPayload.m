function model=v2SelectPayload(model,folder,frames)
% Predeclared six candidates; all selection channels are separate from training/testing.
if nargin<3,frames=8;end
candidates=[0 0;0 .5;0 1;1 0;1 .5;1 1];points=[0 4 8 12 16 20];
totals=zeros(6,3,4);raw=cell(0,8);
for candidate=1:6
 trial=model;
 for scheme=["MR" "ZF" "PZF"],trial.Selection.(scheme)=candidates(candidate,:);end
 for antennas=[32 64]
  cfg=v2Config(antennas);r=muResources(cfg);
  result=v2Benchmark(cfg,r,trial,frames,points,9000000+antennas*1000,["MR" "ZF" "PZF"],3);
  for p=1:3
   e=sum(result.SlotErrors(:,:,p,:,:),'all');bits=numel(result.SlotErrors(:,:,p,:,:))*r.PayloadBits;
   bad=nnz(result.CRCFailed(:,:,p,:,:));blocks=numel(result.CRCFailed(:,:,p,:,:));
   totals(candidate,p,:)=totals(candidate,p,:)+reshape([e bits bad blocks],1,1,4);
   raw(end+1,:)={candidate,antennas,result.Schemes(p),candidates(candidate,1), ...
       candidates(candidate,2),e/bits,bad/blocks,bits}; %#ok<AGROW>
  end
 end
end
for p=1:3
 score=[totals(:,p,1)./totals(:,p,2) totals(:,p,3)./totals(:,p,4)];
 [~,order]=sortrows(score,[1 2]);names=["MR" "ZF" "PZF"];
 model.Selection.(names(p))=candidates(order(1),:);
 fprintf('%s frozen TX/RX multipliers: %.1f %.1f\n',names(p),model.Selection.(names(p)));
end
T=cell2table(raw,'VariableNames',{'Candidate','BS','Precoder','TXMultiplier','RXMultiplier', ...
 'PostLDPC_BER','CRCBlockErrorRate','PayloadBits'});writetable(T,fullfile(folder,'v2_payload_validation.csv'));
model.PayloadValidation=struct('FramesPerArray',frames,'SNR',points,'Candidates',candidates,'Counts',totals);
end
