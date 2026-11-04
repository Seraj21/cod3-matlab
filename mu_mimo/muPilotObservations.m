function samples = muPilotObservations(received,r,direction)
% Return de-spread LS samples, indexed K x L x receive branches x user ports.
R = size(received,3);
U = size(r.SRS,3);
samples = complex(zeros(r.K,r.L,R,U));
if direction=="UL"
    flat = reshape(received,r.K*r.L,R);
    for u = 1:U
        reference = r.SRS(:,:,u);
        indices = find(reference);
        temp = complex(zeros(r.K*r.L,R));
        temp(indices,:) = flat(indices,:)./reference(indices);
        samples(:,:,:,u) = reshape(temp,r.K,r.L,R);
    end
else
    for group = 1:numel(r.DMRSGroups)
        g = r.DMRSGroups{group};
        for t = g.Times(:).'
            for pair = 1:size(g.Pairs,2)
                frequencies = g.Pairs(:,pair);
                P = reshape(r.DMRS(frequencies,t,g.Ports),2,numel(g.Ports));
                Y = reshape(received(frequencies,t,:),2,R);
                estimate = P\Y;
                for port = 1:numel(g.Ports)
                    samples(g.Centers(pair),t,:,g.Ports(port)) = reshape(estimate(port,:),1,1,R);
                end
            end
        end
    end
end
end
