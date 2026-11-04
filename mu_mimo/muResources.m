function r = muResources(cfg)
r.Carrier = nrCarrierConfig('NSizeGrid',cfg.NumRB,'SubcarrierSpacing',60,'CyclicPrefix','extended');
r.OFDM = nrOFDMInfo(r.Carrier);
r.K = 12*cfg.NumRB;
r.L = r.Carrier.SymbolsPerSlot;
r.G = r.K/cfg.PRGWidth;
r.Samples = sum(r.OFDM.SymbolLengths(1:r.L));
r.FrameDuration = r.Samples/r.OFDM.SampleRate;
U = cfg.Users;
pdsch = nrPDSCHConfig('PRBSet',0:cfg.NumRB-1,'SymbolAllocation',[5 7], ...
    'MappingType','B','Modulation','256QAM','NumLayers',U,'EnablePTRS',false);
pdsch.DMRS.DMRSPortSet = 0:U-1;
pdsch.DMRS.NumCDMGroupsWithoutData = ceil(U/2);
pdsch.DMRS.DMRSAdditionalPosition = 3;
pdsch.DMRS.CustomSymbolSet = [5 8 11];
idx = nrPDSCHIndices(r.Carrier,pdsch);
r.DataMask = false(r.K,r.L);
r.DataMask(mod(double(idx(:))-1,r.K*r.L)+1) = true;
idx = nrPDSCHDMRSIndices(r.Carrier,pdsch);
sym = nrPDSCHDMRS(r.Carrier,pdsch);
r.DMRS = complex(zeros(r.K,r.L,U));
r.DMRS(idx) = sym;
r.DownlinkLayout = cell(U,1);
r.DMRSGroups = cell(ceil(U/2),1);
for group = 1:ceil(U/2)
    ports = (group-1)*2+1:min(group*2,U);
    boost = sqrt(U/numel(ports));
    r.DMRS(:,:,ports) = r.DMRS(:,:,ports).*boost;
    [frequencies,times] = find(abs(r.DMRS(:,:,ports(1)))>0);
    times = unique(times);
    frequencies = unique(frequencies);
    pairs = reshape(frequencies,2,[]);
    centers = mean(pairs,1).';
    local = centers(centers<=cfg.PRGWidth);
    noiseGain = 1/(2*boost^2);
    for u = ports
        r.DownlinkLayout{u} = muPilotLayout(cfg.PRGWidth,r.L,local,times,noiseGain,"DL"+group);
    end
    r.DMRSGroups{group} = struct('Ports',ports,'Pairs',pairs,'Centers',centers,'Times',times);
end
r.PTRS = complex(zeros(r.K,r.L,U));
r.PTRSIndices = cell(U,1);
dataTimes = find(any(r.DataMask,1));
for u = 1:U
    frequencies = (0:2:cfg.NumRB-1)*12+(1+2*(u-1));
    mask = false(r.K,r.L);
    mask(frequencies,dataTimes) = true;
    mask = mask&r.DataMask;
    indices = find(mask);
    stream = RandStream('mt19937ar','Seed',2200+u);
    q = (1-2*randi(stream,[0 1],numel(indices),1)) ...
        +1i*(1-2*randi(stream,[0 1],numel(indices),1));
    temp = complex(zeros(r.K,r.L));
    temp(indices) = sqrt(U/2)*q;
    r.PTRS(:,:,u) = temp;
    r.PTRSIndices{u} = indices;
end
r.DataMask(any(abs(r.PTRS)>0,3)) = false;
r.DataIndices = find(r.DataMask);
[~,r.DataTimes] = ind2sub([r.K r.L],r.DataIndices);
r.ReferenceGrid = r.DMRS+r.PTRS;
r.SRS = complex(zeros(r.K,r.L,U));
r.UplinkLayout = cell(U,1);
for u = 1:U
    freq = u:U:r.K;
    mask = false(r.K,r.L);
    mask(freq,cfg.UplinkSymbols) = true;
    indices = find(mask);
    stream = RandStream('mt19937ar','Seed',1200+u);
    temp = complex(zeros(r.K,r.L));
    temp(indices) = ((1-2*randi(stream,[0 1],numel(indices),1)) ...
        +1i*(1-2*randi(stream,[0 1],numel(indices),1)))/sqrt(2);
    r.SRS(:,:,u) = temp;
    r.UplinkLayout{u} = muPilotLayout(cfg.PRGWidth,r.L,u:U:cfg.PRGWidth, ...
        cfg.UplinkSymbols,1,"UL"+u);
end
r.CodedBits = numel(r.DataIndices)*cfg.Qm;
r.PayloadBits = 8*floor(cfg.Rate*r.CodedBits/8);
r.Codec = ExperimentalLDPC(r.PayloadBits,r.CodedBits,cfg.Qm,cfg.Rate);
stream = RandStream('mt19937ar','Seed',1729);
r.BitOrder = randperm(stream,r.CodedBits).';
r.PDSCH = pdsch;
assert(~any(r.DataMask(any(abs(r.DMRS)>0,3))));
assert(~any(r.DataMask(any(abs(r.PTRS)>0,3))));
assert(all(find(any(r.DataMask,1))>=6));
end
