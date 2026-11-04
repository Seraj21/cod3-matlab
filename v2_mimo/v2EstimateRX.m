function [h,variance,observed]=v2EstimateRX(input,model,r,method)
% Every method uses the same unitary receive basis and CRC-selected combiner.
h=complex(zeros(r.K,r.L,16,4));variance=zeros(size(h));
observed=v2BeamObserved(input.Observed,r);
if method=="Perfect"
 for user=1:4
  [~,truth,~,order]=v2RXCanonical(input,r,user);
  h(:,:,(user-1)*4+(1:4),order)=truth;
 end
 return
end
states=cell(4,1);metas=cell(4,1);orders=cell(4,1);X=cell(4,1);
for user=1:4
 [samples,~,layouts,orders{user}]=v2RXCanonical(input,r,user);
 states{user}=v2CEBase(samples,input.Noise((user-1)*4+(1:4)),layouts,model.PriorsRX);
 if method=="CNN",[X{user},~,metas{user}]=v2RXFeatures(states{user},[],layouts,r);end
end
if method=="CNN"
 prediction=double(extractdata(predict(model.NetworkRX,dlarray(cat(4,X{:}),'SSCB'))));
end
for user=1:4
 state=states{user};rows=(user-1)*4+(1:4);
 if method=="LS"
  value=state.LS;v=state.VarianceLS;
 elseif method=="LMMSE"
  value=state.Wiener;v=state.VarianceWiener;
 else
  selected=(user-1)*metas{user}.G+(1:metas{user}.G);
  [update,v]=v2RXPrediction(prediction(:,:,:,selected),state,metas{user});
  value=state.Wiener+model.BlendRX*update;
  v=applyVarianceFactors(v,model.PriorsRX,r.DownlinkLayout,orders{user});
  v=model.BlendRX*v+(1-model.BlendRX)*state.VarianceWiener;
 end
 h(:,:,rows,orders{user})=value;variance(:,:,rows,orders{user})=v;
end
end
function v=applyVarianceFactors(v,priors,layouts,order)
for u=1:4
 role="Other";if u==1,role="Own";end
 for branch=1:4
  key=role+"_"+layouts{order(u)}.Label+"_R"+branch;
  v(:,:,branch,u)=v(:,:,branch,u)*priors.(key).VarianceFactor.CNN;
 end
end
end
