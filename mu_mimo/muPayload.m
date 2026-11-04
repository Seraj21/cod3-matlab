function [grid,payload] = muPayload(cfg,r,seed)
stream = RandStream('mt19937ar','Seed',seed);
payload = int8(randi(stream,[0 1],r.PayloadBits,cfg.Users));
grid = r.ReferenceGrid;
for u = 1:cfg.Users
    coded = r.Codec.encode(payload(:,u));
    temp = grid(:,:,u);
    temp(r.DataIndices) = qammod(coded(r.BitOrder),cfg.M,'InputType','bit','UnitAveragePower',true);
    grid(:,:,u) = temp;
end
end
