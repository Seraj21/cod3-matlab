function [estimate,variance,state] = muCE(samples,noise,layouts,models,r,method,network,blend,groupSize)
arguments
    samples
    noise
    layouts
    models
    r
    method (1,1) string {mustBeMember(method,["LS" "LMMSE" "CNN"])}
    network = []
    blend (1,1) double = 1
    groupSize (1,1) double = 0
end
[K,L,R,U] = size(samples);
W = layouts{1}.W;
G = K/W;
noise = noise(:).';
if isscalar(noise),noise=repmat(noise,1,R);end
assert(numel(noise)==R);
state = struct('LS',complex(zeros(K,L,R,U)),'Wiener',complex(zeros(K,L,R,U)), ...
    'Pilot',samples,'Scale',zeros(G,R,U),'RelativeNoise',zeros(G,R,U), ...
    'VarianceLS',zeros(K,L,R,U),'VarianceWiener',zeros(K,L,R,U));
for u = 1:U
    p = layouts{u};
    model = models.(p.Label);
    patches = reshape(permute(reshape(double(samples(:,:,:,u)),W,G,L,R),[1 3 2 4]),W*L,[]).';
    obs = patches(:,p.Observed);
    n = repelem(noise,G).'*p.NoiseGain;
    scale = sqrt(max(mean(abs(obs).^2,2)-n,max(.1*n,1e-5)));
    relative = n./scale.^2;
    inverse = 1./(model.D+relative);
    ls = obs*p.Linear;
    wiener = (((obs./scale)*model.V).*inverse)*model.Cross.*scale;
    vls = scale.^2.*model.LSBias+n.*model.LSNoise;
    vw = max(0,scale.^2.*(model.YY-inverse*abs(model.Cross).^2));
    state.LS(:,:,:,u) = restoreRows(ls,W,L,G,R);
    state.Wiener(:,:,:,u) = restoreRows(wiener,W,L,G,R);
    state.VarianceLS(:,:,:,u) = restoreRows(vls,W,L,G,R).*model.VarianceFactor.LS;
    state.VarianceWiener(:,:,:,u) = restoreRows(vw,W,L,G,R).*model.VarianceFactor.LMMSE;
    state.Scale(:,:,u) = reshape(scale,G,R);
    state.RelativeNoise(:,:,u) = reshape(relative,G,R);
end
if method=="LS"
    estimate = state.LS;
    variance = state.VarianceLS;
elseif method=="LMMSE"
    estimate = state.Wiener;
    variance = state.VarianceWiener;
else
    assert(~isempty(network));
    if groupSize==0,groupSize=R;end
    [features,meta] = muCNNFeatures(state,layouts,r,groupSize);
    correction = double(extractdata(predict(network,dlarray(features,'SSCB'))));
    correction = muCNNRestore(correction,meta,state.Scale);
    estimate = state.Wiener+blend*correction;
    variance = state.VarianceWiener;
    for u = 1:U
        m = models.(layouts{u}.Label);
        variance(:,:,:,u) = variance(:,:,:,u).*m.VarianceFactor.CNN/max(m.VarianceFactor.LMMSE,eps);
    end
end
end
function h = restoreRows(rows,W,L,G,R)
h = reshape(permute(reshape(rows.',W,L,G,R),[1 3 2 4]),W*G,L,R);
end
