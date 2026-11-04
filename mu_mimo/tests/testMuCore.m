function tests = testMuCore
tests = functiontests(localfunctions);
end
function testCachedPhaseNoiseMatchesSeededReference(testCase)
s = struct('SampleRate',61.44e6,'PhaseNoiseLevel',[-85 -100 -115], ...
    'PhaseNoiseOffset',[1e4 1e5 1e6]);
clear muPhaseNoise
rows = zeros(4,5);
row = 0;
for count = [15360 16098]
    for seed = [321 654]
        row = row+1;
        tic
        reference = comm.PhaseNoise('Level',s.PhaseNoiseLevel, ...
            'FrequencyOffset',s.PhaseNoiseOffset,'SampleRate',s.SampleRate, ...
            'RandomStream','mt19937ar with seed','Seed',seed);
        expected = reference(complex(ones(count,1)));
        referenceTime = toc;
        before = rng;
        tic
        actual = muPhaseNoise(count,s,seed);
        cachedTime = toc;
        verifyEqual(testCase,rng,before);
        verifyEqual(testCase,actual,expected,'AbsTol',1e-13);
        repeated = muPhaseNoise(count,s,seed);
        verifyEqual(testCase,repeated,actual);
        rows(row,:) = [count seed referenceTime cachedTime max(abs(actual-expected))];
    end
end
folder = fullfile(getenv('GITHUB_WORKSPACE'),'mu_probe');
if ~isfolder(folder),mkdir(folder);end
writetable(array2table(rows,'VariableNames',{'Samples','Seed','FreshSeconds','CachedSeconds','MaxError'}), ...
    fullfile(folder,'phase_noise_benchmark.csv'));
end
function testPrecoderConstraints(testCase)
stream = RandStream('mt19937ar','Seed',173);
H = complex(randn(stream,4,8),randn(stream,4,8));
for n = [32 64]
    front = muFrontEnd(n,8,.7,.3);
    for scheme = ["MR" "ZF" "PZF"]
        [B,d] = muPrecoder(H,scheme,front,4,2);
        verifyEqual(testCase,d.TotalPower,1,'AbsTol',1e-12);
        verifyEqual(testCase,d.ColumnPower,.25*ones(1,4),'AbsTol',1e-12);
        if scheme=="ZF"
            E = H*B;
            verifyLessThan(testCase,norm(E-diag(diag(E)),'fro'),1e-10);
        elseif scheme=="PZF"
            verifyLessThan(testCase,d.ConstantModulusError,1e-12);
            verifyLessThanOrEqual(testCase,d.FinalObjective,d.InitialObjective+1e-10);
            phase = angle(front.FRF*B)/(2*pi/16);
            verifyLessThan(testCase,max(abs(phase-round(phase)),[],'all'),1e-12);
        end
    end
end
end
function testRFProjectionCovariance(testCase)
for n = [32 64]
    front = muFrontEnd(n,8,.7,.3);
    actual = front.FRF'*front.PhysicalCorrelation*front.FRF;
    restored = (front.ChannelGain*front.ChannelGain.').*front.Correlation;
    verifyEqual(testCase,actual,restored,'AbsTol',1e-12);
    verifyEqual(testCase,front.FRF'*front.FRF,eye(8),'AbsTol',1e-12);
end
end
function testTDLReciprocity(testCase)
cfg = muConfig(32,NumRB=12,Speed=0);
front = muFrontEnd(32,8,.7,.3);
carrier = nrCarrierConfig('NSizeGrid',12,'SubcarrierSpacing',60,'CyclicPrefix','extended');
o = nrOFDMInfo(carrier);
ch = muChannel(cfg,front,o.SampleRate,122);
N = sum(o.SymbolLengths(1:12))+1000;
[~,Hdl,td] = ch(zeros(N,8,'single'),carrier);
swapTransmitAndReceive(ch);
reset(ch);
[~,Hul,tu] = ch(zeros(N,16,'single'),carrier);
verifyEqual(testCase,tu,td);
verifyEqual(testCase,Hul,permute(Hdl,[1 2 4 3]),'AbsTol',2e-5);
end
