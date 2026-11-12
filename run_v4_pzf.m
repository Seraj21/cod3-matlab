function run_v4_pzf(antennas,frames,folder,snrDb,seedOffset,modelFile,modes,methods,matchedFile,alignedFile,scheme,numRB)
% Paired measurement with a common waveform for receiver-only estimators.
addpath(pwd,'mu_mimo','v2_mimo');
if nargin<4,snrDb=[0:.5:10 11:40];end
if nargin<5,seedOffset=90000000;end
if nargin<6,modelFile='v3_refined/v3_model.mat';end
if nargin<7,modes="None";end
if nargin<8,methods=["LS" "LMMSE" "V3 CNN" "Perfect CSI"];end
if nargin<9,matchedFile="";end
if nargin<10,alignedFile="";end
if nargin<11,scheme="PZF";end
if nargin<12,numRB=66;end
scheme=string(scheme);
assert(isscalar(scheme) && ismember(scheme,["MR" "ZF" "PZF"]));
modes=string(modes);methods=string(methods);
assert(all(ismember(modes,["None" "NMS50" "LBP50" "APP20" "APP50" "CSI-APP20" "CSI-APP50"])));
assert(all(ismember(methods,["LS" "LS on common packet" "LMMSE" "Matched LMMSE" ...
 "V3 CNN" "Matched CNN" "Aligned LMMSE" "Aligned CNN" "Perfect CSIR" "Perfect CSI"])));
assert(all(diff(snrDb)>0) && min(snrDb)>=0 && max(snrDb)<=40);
loaded=load(modelFile,'model');model=loaded.model;
cfg=v3Config(antennas,numRB,"mobile");r=v3Resources(cfg);
selection=model.Selection.(scheme);
assert(model.Version==6 && selection(1)==0);
rx=model.Receivers.(scheme);rx.BlendRX=rx.BlendRX*selection(2);
if any(ismember(methods,["Matched LMMSE" "Matched CNN"]))
 matched=loadReceiver(matchedFile,model,false,scheme,cfg);
end
if any(ismember(methods,["Aligned CNN" "Aligned LMMSE"]))
 aligned=loadReceiver(alignedFile,model,true,scheme,cfg);
end
txPriors=model.PriorsTX;
if exist('aligned','var'),txPriors=aligned.PriorsTX;end
needTruth=any(methods=="Perfect CSIR");
rows=cell(frames*numel(snrDb)*numel(methods)*numel(modes)*cfg.Users,17);at=0;
for frame=1:frames
 scene=v2Sounding(cfg,r,seedOffset+frame);
 [grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
 % Scoring references never enter the channel estimator or the equalizer.
 codedReference=zeros(r.CodedBits,cfg.Users,'int8');
 symbolReference=complex(zeros(numel(r.DataIndices),cfg.Users));
 for user=1:cfg.Users
  codedReference(:,user)=r.Codec.encode(payload(:,user));
  layer=grid(:,:,user);symbolReference(:,user)=layer(r.DataIndices);
 end
 tx=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,txPriors);
 [weights,power]=v3Weights(tx.Wiener,tx.VarianceWiener,scheme,scene,r);
 assert(power.MaximumPowerError<1e-6);
 common=v2Waveform(scene,r,grid,weights,needTruth);
 if needTruth && frame==1
  reference=v2Waveform(scene,r,grid,weights,false);
  assert(isequal(common.Waveform,reference.Waveform),'Requesting truth changed the transmitted experiment');
  clear reference
 end
 if any(methods=="LS")
  weights=v3Weights(tx.LS,tx.VarianceLS,scheme,scene,r);
  lsPacket=v2Waveform(scene,r,grid,weights,false);
 end
 if any(methods=="Perfect CSI")
  weights=v3Weights(scene.ULTruth,zeros(size(scene.ULTruth)),scheme,scene,r);
  oracle=v2Waveform(scene,r,grid,weights,true);
 end
 for snr=snrDb
  commonInput=v3Receive(common,scene,r,snr,false,false);
  oracleChannel=commonInput.Truth;commonInput.Truth=[];
  for method=methods
   input=commonInput;receiver=rx;internal=method;
   switch method
    case "LS"
     input=v3Receive(lsPacket,scene,r,snr,false,false);
    case "LS on common packet"
     internal="LS";
     if exist('aligned','var'),receiver=aligned;end
    case "Perfect CSIR"
     input.Truth=oracleChannel;internal="Perfect";
    case "Perfect CSI"
     input=v3Receive(oracle,scene,r,snr,true,false);internal="Perfect";
    case "V3 CNN"
     internal="CNN";
    case "Matched LMMSE"
     receiver=matched;internal="LMMSE";
    case "Matched CNN"
     receiver=matched;internal="CNN";
    case "Aligned LMMSE"
     receiver=aligned;internal="LMMSE";
    case "Aligned CNN"
     receiver=aligned;internal="CNN";
   end
   [h,v,observed]=v3EstimateRX(input,receiver,r,internal);input.Observed=observed;
   measureRetry=method~="Perfect CSI" && any(modes~="None");
   [errors,failed,detail,primary]=v2Decode(input,h,v,payload,cfg,r);
   preBER=mean((primary.LLR<0)~=codedReference,1);
   evm=100*sqrt(mean(abs(primary.Symbols-symbolReference).^2,1)./mean(abs(symbolReference).^2,1));
   estimatedSINR=-10*log10(max(mean(primary.Variance,1),eps));
   for mode=modes
    e=errors;f=failed;iterations=zeros(1,cfg.Users);
    if measureRetry && mode~="None" && any(failed)
     [decoded,f,iterations]=v4Retry(primary,failed,cfg,r,mode);
     e=sum(decoded~=payload,1);
    end
    for user=1:cfg.Users
     at=at+1;
     rows(at,:)={antennas,scene.Seed,snr,char(method),char(mode),user,r.PayloadBits, ...
      e(user),f(user),errors(user),failed(user),iterations(user),detail.SRSBeamAccepted(user),char(scheme), ...
      preBER(user),evm(user),estimatedSINR(user)};
    end
   end
  end
 end
 if ~isfolder(folder),mkdir(folder);end
 table=cell2table(rows(1:at,:),'VariableNames',{'BS','ChannelSeed','SNR_dB','Estimator','Retry', ...
  'UE','PayloadBits','Errors','CRCFailed','BaselineErrors','BaselineCRCFailed','ExtraIterations','SRSBeamAccepted','Precoder', ...
  'PrimaryPreLDPC_BER','PrimaryEVM_percent','PrimaryEstimatedSINR_dB'});
 writetable(table,fullfile(folder,'pzf_slots.csv'));
 fprintf('%s BS%d channel %d/%d seed%d complete\n',scheme,antennas,frame,frames,scene.Seed);
end
assert(at==size(rows,1));
settings=struct('FrameDuration',r.FrameDuration,'CodeRate',cfg.Rate,'CodedBits',r.CodedBits, ...
 'PayloadBits',r.PayloadBits,'SeedOffset',seedOffset,'SNR_dB',snrDb,'FrameCount',frames,'Config',cfg,'Precoder',char(scheme));
save(fullfile(folder,'pzf_settings.mat'),'settings');
% Emit an auditable high-SNR summary; job success alone is not link success.
for method=methods
 for mode=modes
  t=table(string(table.Estimator)==method & string(table.Retry)==mode & table.SNR_dB==max(snrDb),:);
  correct=nnz(t.Errors==0 & ~t.CRCFailed);
  summary=struct('BS',antennas,'Precoder',scheme,'Estimator',method,'Retry',mode, ...
   'SNR_dB',max(snrDb),'Frames',frames,'BER',sum(t.Errors)/sum(t.PayloadBits), ...
   'CorrectBlocks',correct,'Blocks',height(t),'BLER',1-correct/height(t), ...
   'Goodput_Mbps',correct*r.PayloadBits/(frames*r.FrameDuration*1e6), ...
   'PrimaryPreLDPC_BER',mean(t.PrimaryPreLDPC_BER),'PrimaryEVM_percent',mean(t.PrimaryEVM_percent));
  fprintf('V3_LINK_SUMMARY %s\n',jsonencode(summary));
 end
end
end
function receiver=loadReceiver(file,model,projection,scheme,cfg)
loaded=load(file,'receiver');receiver=loaded.receiver;
rate=cfg.Rate;
% Older PZF receivers inherit the geometry signature before V3 assembly.
expected=model.Signature;expected.TargetCodeRate=rate;
legacy=model.Receivers.(scheme).Signature;legacy.TargetCodeRate=rate;
compatible=isequal(receiver.Signature,expected) || isequal(receiver.Signature,legacy);
assert(compatible,'Receiver geometry signature does not match the frozen model');
assert(receiver.Version==model.Version && receiver.Scheme==scheme && ...
 receiver.ProjectionLoss==projection && receiver.TrainingBandwidthRB==66 && ...
 receiver.PilotMode==string(model.Signature.PilotMode) && ...
 receiver.TrainingCodeRate==rate,'Receiver training conditions do not match deployment');
assert(isfield(receiver,'TrainingChannelConfig'),'Retrain receiver with explicit channel configuration');
assert(isfield(receiver,'PriorsTX'),'Retrain the transmitter prior for this operating point');
assert(isfield(receiver.TrainingChannelConfig,'PrecoderTimeMode'), ...
 'Retrain receiver with explicit pilot/precoder temporal support');
for name=string(fieldnames(receiver.TrainingChannelConfig)).'
 assert(isequal(receiver.TrainingChannelConfig.(name),cfg.(name)), ...
  'Receiver channel configuration mismatch: %s',name);
end
end
