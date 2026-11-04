function v2_phase_diagnostics(bs)
% Bounded diagnosis, not a replacement for the independent frozen-model test.
addpath('mu_mimo','v2_mimo');load('v2_frozen/v2_model.mat','model');
addpath('external_manopt');importmanopt;
rows=cell(0,14);opt=cell(0,8);sources=["LS" "LMMSE" "CNN" "Perfect"];
for rate=[.5 .125]
 cfg=v2Config(bs);cfg.Rate=rate;r=muResources(cfg);
 for frame=1:2
  scene=v2Sounding(cfg,r,6600000+bs*1000+frame);[grid,payload]=muPayload(cfg,r,muSeed(scene.Seed,70));
  state=v2CEBase(scene.ULSamples,scene.ULNoise,r.UplinkLayout,model.PriorsTX);
  rx=model.Receivers.PZF;rx.BlendRX=rx.BlendRX*model.Selection.PZF(2);
  for method=1:4
   if method==4,csit=scene.ULTruth;variance=zeros(size(csit));
   elseif method==1,csit=state.LS;variance=state.VarianceLS;
   else,csit=state.Wiener;variance=state.VarianceWiener;
   end
   % Frozen payload validation selected TX multiplier zero, including CNN.
   assert(model.Selection.PZF(1)==0);
   [baseline,d0]=v2Weights(csit,"PZF",scene,r);
   [robust,d1]=v2RobustPhaseWeights(csit,variance,scene,r,baseline);
   if rate==.125 && frame==1 && method==2
    opt=manoptControl(csit,baseline,robust,d1,scene,r,bs);
   end
   for variant=1:2
    weights=baseline;if variant==2,weights=robust;end
    packet=v2Waveform(scene,r,grid,weights,true);
    for snr=[12 16 20]
     input=v2Receive(packet,scene,r,snr,method==4,false);
     for oracle=[false true]
      receiver=sources(method);if oracle,receiver="Perfect";end
      [h,v,observed]=v2EstimateRX(input,rx,r,receiver);decodedInput=input;decodedInput.Observed=observed;
      [e,f]=v2Decode(decodedInput,h,v,payload,cfg,r);correct=nnz(e==0 & ~f);
      label="Baseline PZF";if variant==2,label="Robust phase-only control";end
      rows(end+1,:)={bs,rate,frame,label,sources(method),oracle,snr,sum(e),numel(payload), ...
       correct,correct*r.PayloadBits/r.FrameDuration/1e6,d0.MaximumPowerError, ...
       d1.MeanInitialObjective,d1.MeanFinalObjective}; %#ok<AGROW>
     end
    end
   end
  end
  fprintf('Completed BS%d rate%.3f phase diagnostic frame%d/2\n',bs,rate,frame);
  T=cell2table(rows,'VariableNames',{'BSAntennas','TargetCodeRate','Frame','Variant','Estimator', ...
   'PerfectRXControl','SNR_dB','Errors','Bits','CorrectBlocks','Goodput_Mbps', ...
   'PowerError','InitialRobustCost','OptimizedRobustCost'});
  writetable(T,'v2_phase_diagnostics.csv');
 end
end
O=cell2table(opt,'VariableNames',{'BSAntennas','Page','User','GradientRelativeError', ...
 'BaselineCost','RobustCost','ManoptContinuousCost','ManoptQuantizedCost'});
writetable(O,'v2_phase_manopt.csv');
fprintf('V2_PHASE_CONTROL_BEGIN\n%s\nV2_PHASE_CONTROL_END\n',jsonencode(table2struct(T)));
end
function rows=manoptControl(h,old,new,d,scene,r,bs)
cfg=scene.Config;times=cfg.DownlinkSymbols;
H=reshape(permute(double(h(:,times,:,:)),[4 3 1 2]),4,8,[]);
a=1/sqrt(32);Aold=reshape(double(old(:,:,:,times)),8,4,[]);Anew=reshape(double(new(:,:,:,times)),8,4,[]);
eta=d.PosteriorPenalty;pages=round(linspace(1,size(H,3),12));rows=cell(0,8);
options=struct('verbosity',0,'maxiter',200,'tolgradnorm',1e-7);
for page=pages
 for user=1:4
  other=setdiff(1:4,user);X=H(:,:,page);A=a^2*(X(other,:)'*X(other,:));B=a^2*(X(user,:)'*X(user,:));n=eta(page);
  cost=@(z)log(max(real(z'*A*z)+n,eps))-log(max(real(z'*B*z),eps));
  gradient=@(z)2*(A*z/(real(z'*A*z)+n)-B*z/max(real(z'*B*z),eps));
  problem=struct('M',complexcirclefactory(8),'cost',cost,'egrad',gradient);
  z=Aold(:,user,page)/a;direction=1i*z.*cos((1:8).');step=1e-6;
  fd=(cost(z.*exp(1i*step*cos((1:8).')))-cost(z.*exp(-1i*step*cos((1:8).'))))/(2*step);
  analytic=real(gradient(z)'*direction);err=abs(fd-analytic)/max([1 abs(fd) abs(analytic)]);
  assert(err<1e-5,'Phase-objective gradient failed finite differences');
  zref=conjugategradient(problem,z,options);zq=exp(1i*2*pi/16*round(angle(zref)/(2*pi/16)));
  zn=Anew(:,user,page)/a;
  rows(end+1,:)={bs,page,user,err,exp(cost(z)),exp(cost(zn)),exp(cost(zref)),exp(cost(zq))}; %#ok<AGROW>
 end
end
end
