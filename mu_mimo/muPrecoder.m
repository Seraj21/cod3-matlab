function [B,diagnostic] = muPrecoder(H,scheme,front,phaseBits,iterations)
% H is users by RF chains. Total radiated power is one, split equally.
arguments
    H double
    scheme (1,1) string {mustBeMember(scheme,["MR" "ZF" "PZF"])}
    front struct
    phaseBits (1,1) double = 4
    iterations (1,1) double = 2
end
[users,rf] = size(H);
assert(users<=rf && size(front.FRF,2)==rf);
if scheme=="MR"
    B = H';
else
    B = pinv(H,1e-10*max(norm(H,2),eps));
end
for u = 1:users
    B(:,u) = B(:,u)/max(norm(front.FRF*B(:,u)),eps)/sqrt(users);
end
initial = 0;
final = 0;
if scheme=="PZF"
    amplitude = 1/sqrt(users*rf);
    step = 2*pi/2^phaseBits;
    B = amplitude*exp(1i*step*round(angle(B)/step));
    phases = amplitude*exp(1i*step*(0:2^phaseBits-1));
    for u = 1:users
        other = setdiff(1:users,u);
        eta = 1e-6*max(mean(abs(H).^2,'all'),eps);
        w = B(:,u);
        response = H*w;
        initial = initial+(sum(abs(response(other)).^2)+eta)/max(abs(response(u))^2,eps);
        for iteration = 1:iterations
            for q = 1:rf
                candidates = response+H(:,q)*(phases-w(q));
                cost = (sum(abs(candidates(other,:)).^2,1)+eta)./max(abs(candidates(u,:)).^2,eps);
                currentCost = (sum(abs(response(other)).^2)+eta)/max(abs(response(u))^2,eps);
                [best,index] = min(cost);
                if best < currentCost-1e-14*max(1,currentCost)
                    w(q) = phases(index);
                    response = candidates(:,index);
                end
            end
        end
        B(:,u) = w;
        final = final+(sum(abs(response(other)).^2)+eta)/max(abs(response(u))^2,eps);
    end
end
physical = front.FRF*B;
response = H*B;
diagonal = diag(response);
leak = response-diag(diagonal);
diagnostic = struct('TotalPower',sum(abs(physical).^2,'all'), ...
    'ColumnPower',sum(abs(physical).^2,1),'LeakageRatio',sum(abs(leak).^2,'all')/max(sum(abs(diagonal).^2),eps), ...
    'ConditionNumber',cond(H),'InitialObjective',initial,'FinalObjective',final, ...
    'ConstantModulusError',max(abs(abs(physical)-1/sqrt(numel(physical))),[],'all'));
assert(abs(diagnostic.TotalPower-1)<1e-9);
end
