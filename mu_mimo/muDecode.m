function [errors,failed,details] = muDecode(input,h,channelVariance,payload,cfg,r)
% Each UE applies interference-aware MMSE combining using all layer pilots.
errors = zeros(1,cfg.Users);
failed = false(1,cfg.Users);
evm = zeros(1,cfg.Users);
for u = 1:cfg.Users
    rows = (u-1)*cfg.UERx+(1:cfg.UERx);
    G = permute(h(:,:,rows,:),[3 4 1 2]);
    Y = reshape(permute(input.Observed(:,:,rows),[3 1 2]),cfg.UERx,1,r.K,r.L);
    uncertainty = sum(channelVariance(:,:,rows,:),4);
    covariance = pagemtimes(G,'none',G,'ctranspose');
    for branch = 1:cfg.UERx
        n = input.Noise(rows(branch))+uncertainty(:,:,branch);
        covariance(branch,branch,:,:) = covariance(branch,branch,:,:) ...
            +reshape(n,1,1,r.K,r.L);
    end
    desired = G(:,u,:,:);
    w = pagemldivide(covariance,desired);
    alpha = pagemtimes(w,'ctranspose',desired,'none');
    alphaPower = max(abs(alpha).^2,eps);
    z = pagemtimes(w,'ctranspose',Y,'none')./max(real(alpha),eps);
    response = pagemtimes(w,'ctranspose',G,'none');
    response(:,u,:,:) = 0;
    interference = sum(abs(response).^2,2);
    noisePower = zeros(1,1,r.K,r.L);
    for branch = 1:cfg.UERx
        n = input.Noise(rows(branch))+uncertainty(:,:,branch);
        noisePower = noisePower+abs(w(branch,1,:,:)).^2.*reshape(n,1,1,r.K,r.L);
    end
    softVariance = reshape((interference+noisePower)./alphaPower,r.K,r.L);
    equalized = reshape(z,r.K,r.L);
    reference = r.PTRS(:,:,u);
    indices = r.PTRSIndices{u};
    [~,times] = ind2sub([r.K r.L],indices);
    phasor = accumarray(times,equalized(indices).*conj(reference(indices)),[r.L 1]);
    valid = find(abs(phasor)>0);
    if numel(valid)>1
        phase = interp1(valid,unwrap(angle(phasor(valid))), ...
            min(valid(end),max(valid(1),(1:r.L).')),'linear');
        equalized = equalized.*exp(-1i*phase.');
    end
    dmrs = r.DMRS(:,:,u);
    mask = abs(dmrs)>0;
    times = find(any(mask,1));
    residual = zeros(numel(times),1);
    for j = 1:numel(times)
        l = times(j);
        frequencies = find(mask(:,l));
        difference = (abs(equalized(frequencies,l)-dmrs(frequencies,l)).^2 ...
            -softVariance(frequencies,l))./abs(dmrs(frequencies,l)).^2;
        residual(j) = max(0,median(difference));
    end
    residual = interp1(times,residual,min(times(end),max(times(1),(1:r.L).')),'linear');
    variance = max(1e-7,softVariance(r.DataIndices)+residual(r.DataTimes));
    llr = qamdemod(equalized(r.DataIndices),cfg.M,'OutputType','approxllr', ...
        'UnitAveragePower',true,'NoiseVariance',1);
    llr = reshape(reshape(llr,cfg.Qm,[])./reshape(variance,1,[]),[],1);
    restored = zeros(r.CodedBits,1);
    restored(r.BitOrder) = llr;
    [decoded,failed(u)] = r.Codec.decode(max(-50,min(50,restored)));
    errors(u) = nnz(decoded~=payload(:,u));
    evm(u) = mean(variance);
end
details = struct('MeanEffectiveVariance',evm);
end
