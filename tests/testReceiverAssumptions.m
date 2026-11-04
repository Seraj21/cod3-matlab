function tests = testReceiverAssumptions
tests = functiontests(localfunctions);
end

function testPracticalTimingFindsKnownDelay(testCase)
carrier = nrCarrierConfig('NSizeGrid',66,'SubcarrierSpacing',60,'CyclicPrefix','extended');
grid = zeros(12*carrier.NSizeGrid,carrier.SymbolsPerSlot);
indices = sub2ind(size(grid),(1:2:size(grid,1)).',3*ones(size(grid,1)/2,1));
stream = RandStream('mt19937ar','Seed',843);
symbols = exp(1i*pi/2*randi(stream,[0 3],numel(indices),1));
grid(indices) = symbols;
tx = nrOFDMModulate(carrier,grid,'Windowing',0);
delay = 35;
waveform = [zeros(delay,1);tx;zeros(100,1)];
[offset,mag] = nrTimingEstimate(carrier,waveform,indices,symbols);
[~,peak] = max(sum(abs(mag(1:101,:)).^2,2));
verifyEqual(testCase,double(offset),delay);
verifyEqual(testCase,peak-1,delay);
end

function testFftWindowShiftMatchesChannelLabelPhase(testCase)
carrier = nrCarrierConfig('NSizeGrid',66,'SubcarrierSpacing',60,'CyclicPrefix','extended');
info = nrOFDMInfo(carrier);
K = 12*carrier.NSizeGrid;
stream = RandStream('mt19937ar','Seed',1703);
grid = complex(randn(stream,K,carrier.SymbolsPerSlot),randn(stream,K,carrier.SymbolsPerSlot));
tx = nrOFDMModulate(carrier,grid,'Windowing',0);
delta = 3;
shifted = nrOFDMDemodulate(carrier,[tx(1+delta:end);zeros(delta,1)]);
k = (-K/2:K/2-1).';
expected = grid.*exp(1i*2*pi*k*delta/info.Nfft);
verifyLessThan(testCase,max(abs(shifted-expected),[],'all'),1e-8);
end
