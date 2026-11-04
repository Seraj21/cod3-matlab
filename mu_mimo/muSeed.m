function seed = muSeed(base,domain,user)
arguments
    base (1,1) double
    domain (1,1) double
    user (1,1) double = 0
end
seed = 1+mod(104729*base+13007*domain+8191*user,2^32-2);
end
