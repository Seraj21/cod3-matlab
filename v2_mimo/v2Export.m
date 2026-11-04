function v2Export(result,model,folder)
if ~isfolder(folder),mkdir(folder);end
N=result.NumPackets;S=numel(result.SNR);P=numel(result.Schemes);M=numel(result.Estimators);
bitErrors=reshape(sum(sum(result.SlotErrors,1),5),S,P,M);
totalBits=repmat(N*4*result.PayloadBits,size(bitErrors));ber=bitErrors./totalBits;
success=reshape(sum(sum(~result.CRCFailed & result.SlotErrors==0,1),5),S,P,M);
goodput=success.*result.PayloadBits/(N*result.FrameDuration)/1e6;
maximumGoodput=4*result.PayloadBits/result.FrameDuration/1e6;
assert(all(goodput>=0 & goodput<=maximumGoodput+1e-9,'all'));
rows=cell(S*P*M,9);at=0;
for p=1:P
 for m=1:M
  for s=1:S
   at=at+1;rows(at,:)={result.Config.Antennas,result.Schemes(p),result.Estimators(m), ...
    result.SNR(s),ber(s,p,m),bitErrors(s,p,m),totalBits(s,p,m),goodput(s,p,m),success(s,p,m)};
  end
 end
end
metrics=cell2table(rows,'VariableNames',{'BSAntennas','Precoder','Estimator','SNR_dB', ...
 'PostLDPC_BER','PayloadBitErrors','PayloadBits','Goodput_Mbps','CorrectTransportBlocks'});
writetable(metrics,fullfile(folder,'mu_metrics.csv'));
[frame,snr,precoder,estimator,user]=ndgrid(1:N,1:S,1:P,1:M,1:4);
schemes=result.Schemes(:);names=result.Estimators(:);points=result.SNR(:);
slots=table(frame(:),user(:),points(snr(:)),schemes(precoder(:)),names(estimator(:)), ...
 result.SlotErrors(:),result.CRCFailed(:),repmat(result.PayloadBits,numel(frame),1), ...
 result.SRSBeamAccepted(:),'VariableNames',{'Packet','User','SNR_dB','Precoder','Estimator', ...
 'PayloadBitErrors','CRCFailed','PayloadBits','SRSBeamAccepted'});
writetable(slots,fullfile(folder,'mu_slot_errors.csv'));
nets={model.NetworkTX,model.Receivers.MR.NetworkRX,model.Receivers.ZF.NetworkRX,model.Receivers.PZF.NetworkRX};
cnnFingerprint={};
for k=1:numel(nets)
 values=cellfun(@(x)gather(extractdata(x)),nets{k}.Learnables.Value,'UniformOutput',false);
 cnnFingerprint=[cnnFingerprint;values]; %#ok<AGROW>
end
save(fullfile(folder,'mu_results.mat'),'result','bitErrors','totalBits','ber','success','goodput', ...
 'maximumGoodput','cnnFingerprint','-v7');
end
