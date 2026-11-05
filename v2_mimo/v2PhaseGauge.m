function [weights,diagnostic]=v2PhaseGauge(input,h,scene,r,mode)
% An inference-only stream phase reference derived from the estimated channel.
% "quantized" retains the absolute 4-bit lattice. "continuous" retains all
% relative phase-shifter differences and adds one common digital phase/stream.
cfg=scene.Config;W=cfg.BeamWidth;G=r.K/W;times=cfg.DownlinkSymbols;
assert(any(string(mode)==["quantized" "continuous"]));
block=reshape(double(h(:,times,:,:)),W,G,numel(times),cfg.RFChains,cfg.Users);
H=reshape(permute(mean(block,1),[5 4 2 3 1]),cfg.Users,cfg.RFChains,[]);
B=mean(reshape(double(input(:,:,:,times)),cfg.RFChains,cfg.Users,W,G,numel(times)),3);
B=reshape(B,cfg.RFChains,cfg.Users,[]);
amplitude=1/sqrt(cfg.Users*cfg.RFChains);step=2*pi/2^cfg.PhaseBits;
B=amplitude*exp(1i*step*round(angle(B)/step));
before=pagemtimes(H,B);initialB=B;
for user=1:cfg.Users
 phase=angle(before(user,user,:));
 if mode=="quantized",phase=step*round(phase/step);end
 B(:,user,:)=B(:,user,:).*exp(-1i*phase);
end
after=pagemtimes(H,B);
responseError=max(abs(abs(after).^2-abs(before).^2),[],'all')/max(1,max(abs(before).^2,[],'all'));
powerError=max(abs(sum(abs(B).^2,[1 2])-1),[],'all');
relative=B./B(1,:,:);relativeError=max(abs(relative-exp(1i*step*round(angle(relative)/step))),[],'all');
assert(responseError<1e-11 && powerError<1e-11 && relativeError<1e-11);
if mode=="quantized"
 assert(max(abs(B-amplitude*exp(1i*step*round(angle(B)/step))),[],'all')<1e-11);
end
B=reshape(B,cfg.RFChains,cfg.Users,G,numel(times));
weights=complex(zeros(size(input),'single'));
for t=1:numel(times),weights(:,:,:,times(t))=single(repelem(B(:,:,:,t),1,1,W));end
diagnostic=struct('Mode',string(mode),'ResponsePowerRelativeError',responseError, ...
 'TotalPowerError',powerError,'RelativePhaseLatticeError',relativeError, ...
 'MeanWeightChange',mean(abs(reshape(B,size(initialB))-initialB).^2,'all'));
end
