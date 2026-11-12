classdef ExperimentalLDPC
    % NR CRC and LDPC with RV=0..3 circular rate matching for arbitrary Qm.
    % Qm=9 extends the bit interleaver for experimental 512-QAM.
    properties (SetAccess=private)
        PayloadBits
        CodedBits
        Qm
        Info
        Selection
        Permutation
        BlockLengths
        FillerMask
    end
    methods
        function obj = ExperimentalLDPC(payloadBits,codedBits,qm,rate)
            arguments
                payloadBits (1,1) double {mustBeInteger,mustBePositive}
                codedBits (1,1) double {mustBeInteger,mustBePositive}
                qm (1,1) double {mustBeInteger,mustBePositive}
                rate (1,1) double {mustBePositive,mustBeLessThan(rate,1)}
            end
            assert(mod(codedBits,qm)==0);
            obj.PayloadBits = payloadBits;
            obj.CodedBits = codedBits;
            obj.Qm = qm;
            obj.Info = nrDLSCHInfo(payloadBits,rate);
            attached = nrCRCEncode(zeros(payloadBits,1,'int8'),obj.Info.CRC);
            blocks = nrCodeBlockSegmentLDPC(attached,obj.Info.BGN);
            template = nrLDPCEncode(blocks,obj.Info.BGN);
            obj.FillerMask = template == -1;
            C = size(template,2);
            symbols = codedBits/qm;
            obj.BlockLengths = qm*(floor(symbols/C)+(0:C-1 >= C-mod(symbols,C)));
            obj.Selection = cell(C,1);
            obj.Permutation = cell(C,1);
            for block = 1:C
                E = obj.BlockLengths(block);
                valid = find(~obj.FillerMask(:,block));
                obj.Selection{block} = valid(mod((0:E-1).',numel(valid))+1);
                obj.Permutation{block} = reshape(reshape(1:E,E/qm,qm).',[],1);
            end
            assert(sum(obj.BlockLengths)==codedBits);
        end
        function coded = encode(obj,payload,rv)
            if nargin < 3, rv = 0; end
            selection = obj.selectionForRV(rv);
            assert(numel(payload)==obj.PayloadBits);
            attached = nrCRCEncode(int8(payload(:)),obj.Info.CRC);
            segments = nrCodeBlockSegmentLDPC(attached,obj.Info.BGN);
            mother = nrLDPCEncode(segments,obj.Info.BGN);
            coded = zeros(obj.CodedBits,1,'int8');
            first = 1;
            for block = 1:numel(obj.Selection)
                E = obj.BlockLengths(block);
                selected = mother(selection{block},block);
                coded(first:first+E-1) = selected(obj.Permutation{block});
                first = first+E;
            end
        end
        function recovered = recover(obj,llr,rv)
            if nargin < 3, rv = 0; end
            selection = obj.selectionForRV(rv);
            assert(numel(llr)==obj.CodedBits && all(isfinite(llr(:))));
            recovered = zeros(size(obj.FillerMask));
            first = 1;
            for block = 1:numel(selection)
                E = obj.BlockLengths(block);
                selectedLLR = zeros(E,1);
                selectedLLR(obj.Permutation{block}) = double(llr(first:first+E-1));
                recovered(:,block) = accumarray(selection{block},selectedLLR, ...
                    [size(recovered,1) 1]);
                first = first+E;
            end
            % Keep buffers finite; known filler bits become Inf only at decoding.
            recovered(obj.FillerMask) = 0;
        end
        function [payload,crcFailed,iterations,buffer] = decode(obj,llr,maxIterations,algorithm,rv,prior)
            if nargin < 3, maxIterations = 20; end
            if nargin < 4, algorithm = 'Normalized min-sum'; end
            if nargin < 5, rv = 0; end
            if nargin < 6, prior = []; end
            buffer = obj.recover(llr,rv);
            if ~isempty(prior)
                assert(isequal(size(prior),size(buffer)) && all(isfinite(prior(:))), ...
                    'HARQ buffer must match this transport block and contain finite LLRs');
                buffer = buffer+prior;
            end
            recovered = buffer;
            recovered(obj.FillerMask) = Inf;
            [segments,iterations] = nrLDPCDecode(recovered,obj.Info.BGN,maxIterations, ...
                'Algorithm',algorithm);
            [attached,blockErrors] = nrCodeBlockDesegmentLDPC(segments,obj.Info.BGN, ...
                obj.PayloadBits+obj.Info.L);
            [payload,transportError] = nrCRCDecode(attached,obj.Info.CRC);
            crcFailed = any(blockErrors(:)) || any(transportError(:));
        end
        function selection = selectionForRV(obj,rv)
            assert(isscalar(rv) && ismember(rv,0:3),'RV must be 0, 1, 2 or 3');
            if rv==0, selection=obj.Selection; return; end
            % TS 38.212 Table 5.4.2.1-2, full circular buffer (Ncb=N).
            if obj.Info.BGN==1
                offsets=[0 17 33 56]; columns=66;
            else
                offsets=[0 13 25 43]; columns=50;
            end
            N=size(obj.FillerMask,1);Zc=N/columns;
            assert(Zc==floor(Zc));
            order=mod((0:N-1).'+offsets(rv+1)*Zc,N)+1;
            selection=cell(size(obj.Selection));
            for block=1:numel(selection)
                valid=order(~obj.FillerMask(order,block));
                E=obj.BlockLengths(block);
                selection{block}=valid(mod((0:E-1).',numel(valid))+1);
            end
        end
    end
end
