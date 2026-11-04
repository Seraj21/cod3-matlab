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
if ~isfolder('mu_probe'),mkdir('mu_probe');end
writetable(array2table(rows,'VariableNames',{'Samples','Seed','FreshSeconds','CachedSeconds','MaxError'}), ...
    fullfile('mu_probe','phase_noise_benchmark.csv'));
end
