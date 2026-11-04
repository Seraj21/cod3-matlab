function model = muSelectBER(cfg,r,model,frames,folder)
% Predeclared selection uses held-out post-LDPC payload BER, then CRC failures.
points = [0 4 8 12 16 20];
if frames<4,points=[0 10 20];end
multipliers = [0 .5 1];
records = zeros(numel(multipliers),4);
for i = 1:numel(multipliers)
    candidate = model;
    candidate.BlendMultiplier = multipliers(i);
    totalErrors=0;totalBits=0;totalFailed=0;totalBlocks=0;
    for array = [32 64]
        cfg.Antennas=array;
        count=max(1,ceil(frames/2));
        result=muBenchmark(cfg,r,candidate,count,points,3000000+1000*array, ...
            ["MR" "ZF" "PZF"],3);
        totalErrors=totalErrors+sum(result.SlotErrors,'all');
        totalBits=totalBits+numel(result.SlotErrors)*r.PayloadBits;
        totalFailed=totalFailed+nnz(result.CRCFailed);
        totalBlocks=totalBlocks+numel(result.CRCFailed);
    end
    records(i,:)=[multipliers(i) totalErrors/totalBits totalFailed/totalBlocks totalBits];
end
[~,order]=sortrows(records(:,2:3),[1 2]);
model.BlendMultiplier=records(order(1),1);
model.BERSelection=records;
model.BERValidationFrames=2*max(1,ceil(frames/2));
writetable(array2table(records,'VariableNames',{'CNNResidualMultiplier','PostLDPC_BER','CRC_BlockErrorRate','PayloadBits'}), ...
    fullfile(folder,'validation_payload_ber.csv'));
fprintf('Frozen post-LDPC validation multiplier %.2f\n',model.BlendMultiplier);
end
