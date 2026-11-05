function priors=v3LSMoments(priors,layouts)
% Recompute interpolation MSE/noise from the same stored training moments.
% No channel realization or payload from the independent test is consulted.
for user=1:4
 p=layouts{user};L=p.Linear;
 for role=["Own" "Other"]
  for branch=1:4
   key=role+"_"+p.Label+"_R"+branch;m=priors.(key);
   R=(m.V.*m.D)*m.V';C=m.V*m.Cross;
   bias=m.YY+real(sum(conj(L).*(R*L),1))-2*real(sum(conj(L).*C,1)) ...
      +abs(m.MeanY-m.MeanX*L).^2;
   m.LSBias=max(0,bias);m.LSNoise=sum(abs(L).^2,1);m.VarianceFactor.LS=1;priors.(key)=m;
  end
 end
end
end
