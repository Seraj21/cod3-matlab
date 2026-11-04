function tests = testPracticalHybrid
tests = functiontests(localfunctions);
end
function testRateMatcherAgainstNR256QAM(testCase)
stream = RandStream('mt19937ar','Seed',887);
for A = [1000 20000]
    G = 8*ceil(2*A/8);
    codec = ExperimentalLDPC(A,G,8,.5);
    payload = int8(randi(stream,[0 1],A,1));
    attached = nrCRCEncode(payload,codec.Info.CRC);
    blocks = nrCodeBlockSegmentLDPC(attached,codec.Info.BGN);
    mother = nrLDPCEncode(blocks,codec.Info.BGN);
    reference = nrRateMatchLDPC(mother,G,0,'256QAM',1);
    verifyEqual(testCase,codec.encode(payload),reference);
    llr = 12*(1-2*double(reference));
    [decoded,failed] = codec.decode(llr);
    verifyFalse(testCase,failed);
    verifyEqual(testCase,decoded,payload);
end
end
function testNineBitInterleaverRoundTrip(testCase)
stream = RandStream('mt19937ar','Seed',901);
for A = [2048 28512]
    G = 9*ceil(2*A/9);
    codec = ExperimentalLDPC(A,G,9,.5);
    payload = int8(randi(stream,[0 1],A,1));
    coded = codec.encode(payload);
    symbols = qammod(coded,512,'InputType','bit','UnitAveragePower',true);
    llr = qamdemod(symbols,512,'OutputType','approxllr','UnitAveragePower',true, ...
        'NoiseVariance',1e-3);
    [decoded,failed] = codec.decode(max(-50,min(50,llr)));
    verifyFalse(testCase,failed);
    verifyEqual(testCase,decoded,payload);
end
end
function testPowerConstrainedHybridOptimum(testCase)
rng(901);
FRF = exp(1i*randn(12,4))/sqrt(12);
WRF = exp(1i*randn(10,3))/sqrt(10);
H = complex(randn(10,12,6),randn(10,12,6));
measured = pagemtimes(WRF',pagemtimes(H,FRF));
w = hybridPrecoder(FRF,WRF,measured,6);
effective = pagemtimes(w.Receiver',pagemtimes(H,FRF));
gram = mean(pagemtimes(effective,'ctranspose',effective,'none'),3);
[~,D] = eig((gram+gram')/2,FRF'*FRF);
optimum = max(real(diag(D)));
achieved = real(w.FBB(:,1)'*gram*w.FBB(:,1));
verifyEqual(testCase,achieved,optimum,'RelTol',1e-9);
verifyEqual(testCase,w.Precoder,FRF*w.FBB,'AbsTol',1e-12);
verifyEqual(testCase,w.Receiver'*w.Receiver,eye(3),'AbsTol',1e-12);
end
function testPhaseQuantizationAndUPAOrder(testCase)
for rows = [4 8]
    [banks,W,directions] = hybridCodebook(rows,8,4);
    verifySize(testCase,W,[rows*8 8]);
    [m,n] = ndgrid(0:rows-1,0:7);
    a = exp(1i*pi*(m(:)*directions(3,2)+n(:)*directions(3,1)))/sqrt(rows*8);
    verifyEqual(testCase,banks{1}(:,3),a,'AbsTol',1e-12);
    verifyEqual(testCase,banks{1}'*banks{1},eye(8),'AbsTol',1e-12);
    verifyLessThan(testCase,max(abs(angle(banks{1}(:))/(2*pi/16) ...
        -round(angle(banks{1}(:))/(2*pi/16)))),1e-12);
end
end
function testCSIRSMapping(testCase)
carrier = nrCarrierConfig('NSizeGrid',12,'SubcarrierSpacing',60,'CyclicPrefix','extended');
csirs = nrCSIRSConfig('CSIRSType','nzp','RowNumber',6,'Density','one', ...
    'SubcarrierLocations',[0 2 4 6],'SymbolLocations',0,'NumRB',12);
indices = nrCSIRSIndices(carrier,csirs);
symbols = nrCSIRS(carrier,csirs)/sqrt(2);
grid = nrResourceGrid(carrier,8);
grid(indices) = symbols;
verifyEqual(testCase,csirs.NumCSIRSPorts,8);
[estimate,noise] = nrChannelEstimate(carrier,grid,indices,symbols,'CDMLengths',[2 1]);
verifySize(testCase,estimate,[144 12 8 8]);
verifyLessThan(testCase,noise,1e-8);
end
function testCFOFromCyclicPrefix(testCase)
carrier = nrCarrierConfig('NSizeGrid',12,'SubcarrierSpacing',60,'CyclicPrefix','extended');
ofdm = nrOFDMInfo(carrier);
rng(442);
grid = complex(randn(144,12,3),randn(144,12,3));
tx = nrOFDMModulate(carrier,grid,'Windowing',0);
delay = 35;
waveform = [zeros(delay,3);tx;zeros(100,3)];
waveform = waveform.*exp(1i*2*pi*2800*(0:size(waveform,1)-1).'/ofdm.SampleRate);
estimate = estimateCFO(waveform,delay,ofdm,12);
verifyEqual(testCase,estimate,2800,'AbsTol',1e-6);
end
function testCodedAWGNControl(testCase)
stream = RandStream('mt19937ar','Seed',418);
codec = ExperimentalLDPC(2048,4500,9,.5);
payload = int8(randi(stream,[0 1],2048,1));
coded = codec.encode(payload);
tx = qammod(coded,512,'InputType','bit','UnitAveragePower',true);
baseNoise = complex(randn(stream,size(tx)),randn(stream,size(tx)))/sqrt(2);
errors = zeros(1,2);
points = [0 30];
for i = 1:2
    variance = 10^(-points(i)/10);
    rx = tx+sqrt(variance)*baseNoise;
    llr = qamdemod(rx,512,'OutputType','approxllr','UnitAveragePower',true, ...
        'NoiseVariance',variance);
    decoded = codec.decode(max(-50,min(50,llr)));
    errors(i) = nnz(decoded~=payload);
end
verifyGreaterThan(testCase,errors(1),0);
verifyEqual(testCase,errors(2),0);
end
function testExternalRayChannelConjugationAndDelay(testCase)
carrier = nrCarrierConfig('NSizeGrid',12,'SubcarrierSpacing',60,'CyclicPrefix','extended');
ofdm = nrOFDMInfo(carrier);
rng(718);
layer = complex(randn(144,12),randn(144,12));
f = exp(1i*(0:4).')/sqrt(5);
grid = layer.*reshape(f,1,1,5);
tx = nrOFDMModulate(carrier,grid,'Windowing',0);
rays = struct('Receive',[1;1i;-.2],'Transmit',f,'Delays',.37/ofdm.SampleRate,'DopplerFactor',0);
channel = struct('Rays',rays,'InitialTime',0,'Doppler',0);
[rx,H,offset] = applyMIMOChannel(channel,[tx;zeros(100,5)],carrier);
received = nrOFDMDemodulate(carrier,rx(offset+(1:size(tx,1)),:));
effective = pagemtimes(permute(H,[3 4 1 2]),f);
expected = layer.*reshape(permute(effective,[3 4 1 2]),144,12,3);
verifyEqual(testCase,received,double(expected),'AbsTol',2e-6);
end
