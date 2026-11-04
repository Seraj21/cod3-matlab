function muExport(result,model,folder)
if ~isfolder(folder),mkdir(folder);end
config=result.Config;
N=result.NumPackets;S=numel(result.SNR);P=numel(result.Schemes);M=numel(result.Estimators);U=config.Users;
bitErrors=reshape(sum(sum(result.SlotErrors,1),5),S,P,M);
totalBits=repmat(N*U*result.PayloadBits,size(bitErrors));
ber=bitErrors./totalBits;
correct=(~result.CRCFailed)&(result.SlotErrors==0);
success=reshape(sum(sum(correct,1),5),S,P,M);
goodput=success.*result.PayloadBits./(N*result.FrameDuration)/1e6;
maximumGoodput=U*result.PayloadBits/result.FrameDuration/1e6;
assert(all(goodput<=maximumGoodput+1e-9,'all'));
assert(all(result.SlotErrors>=0 & result.SlotErrors<=result.PayloadBits,'all'));
rows=cell(S*P*M,9);
i=0;
for p=1:P
 for m=1:M
  for s=1:S
   i=i+1;
   rows(i,:)={config.Antennas,result.Schemes(p),result.Estimators(m),result.SNR(s), ...
       ber(s,p,m),bitErrors(s,p,m),totalBits(s,p,m),goodput(s,p,m),success(s,p,m)};
  end
 end
end
metrics=cell2table(rows,'VariableNames',{'BSAntennas','Precoder','Estimator','SNR_dB', ...
    'PostLDPC_BER','PayloadBitErrors','PayloadBits','Goodput_Mbps','CorrectTransportBlocks'});
writetable(metrics,fullfile(folder,'mu_metrics.csv'));
[packet,snr,precoder,estimator,user]=ndgrid(1:N,1:S,1:P,1:M,1:U);
schemes=result.Schemes(:);names=result.Estimators(:);points=result.SNR(:);
slots=table(packet(:),user(:),points(snr(:)),schemes(precoder(:)),names(estimator(:)), ...
    result.SlotErrors(:),result.CRCFailed(:), ...
    repmat(result.PayloadBits,numel(packet),1),'VariableNames', ...
    {'Packet','User','SNR_dB','Precoder','Estimator','PayloadBitErrors','CRCFailed','PayloadBits'});
writetable(slots,fullfile(folder,'mu_slot_errors.csv'));
networks={model.NetworkTX,model.NetworkRX};
cnnFingerprint={};
for k=1:2
 values=cellfun(@(x)gather(extractdata(x)),networks{k}.Learnables.Value,'UniformOutput',false);
 cnnFingerprint=[cnnFingerprint;values]; %#ok<AGROW>
end
save(fullfile(folder,'mu_results.mat'),'result','bitErrors','totalBits','ber','success','goodput', ...
    'maximumGoodput','cnnFingerprint','-v7');
end
