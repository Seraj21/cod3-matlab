function h = muLS(samples,layouts)
[K,L,R,U] = size(samples);
W = layouts{1}.W;
G = K/W;
h = complex(zeros(K,L,R,U));
for u = 1:U
 p = layouts{u};
 rows = reshape(permute(reshape(double(samples(:,:,:,u)),W,G,L,R),[1 3 2 4]),W*L,[]).';
 rows = rows(:,p.Observed)*p.Linear;
 h(:,:,:,u) = reshape(permute(reshape(rows.',W,L,G,R),[1 3 2 4]),K,L,R);
end
end
