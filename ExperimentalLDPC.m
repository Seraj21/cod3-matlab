classdef ExperimentalLDPC
    % NR CRC and LDPC with RV=0 circular rate matching for arbitrary Qm.
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
        function coded = encode(obj,payload)
            assert(numel(payload)==obj.PayloadBits);
            attached = nrCRCEncode(int8(payload(:)),obj.Info.CRC);
            segments = nrCodeBlockSegmentLDPC(attached,obj.Info.BGN);
            mother = nrLDPCEncode(segments,obj.Info.BGN);
            coded = zeros(obj.CodedBits,1,'int8');
            first = 1;
            for block = 1:numel(obj.Selection)
                E = obj.BlockLengths(block);
                selected = mother(obj.Selection{block},block);
                coded(first:first+E-1) = selected(obj.Permutation{block});
                first = first+E;
            end
        end
        function [payload,crcFailed,iterations] = decode(obj,llr,maxIterations,algorithm)
            if nargin < 3, maxIterations = 20; end
            if nargin < 4, algorithm = 'Normalized min-sum'; end
            assert(numel(llr)==obj.CodedBits);
            recovered = zeros(size(obj.FillerMask));
            first = 1;
            for block = 1:numel(obj.Selection)
                E = obj.BlockLengths(block);
                selectedLLR = zeros(E,1);
                selectedLLR(obj.Permutation{block}) = double(llr(first:first+E-1));
                recovered(:,block) = accumarray(obj.Selection{block},selectedLLR, ...
                    [size(recovered,1) 1]);
                first = first+E;
            end
            recovered(obj.FillerMask) = Inf;
            [segments,iterations] = nrLDPCDecode(recovered,obj.Info.BGN,maxIterations, ...
                'Algorithm',algorithm);
            [attached,blockErrors] = nrCodeBlockDesegmentLDPC(segments,obj.Info.BGN, ...
                obj.PayloadBits+obj.Info.L);
            [payload,transportError] = nrCRCDecode(attached,obj.Info.CRC);
            crcFailed = any(blockErrors(:)) || any(transportError(:));
        end
    end
end
