function [h,v,observed]=v3EstimateRX(input,receiver,r,method)
% Retain the original learned feature operator for LMMSE/CNN.
% Only the standalone LS comparator may use its independently selected operator.
if method=="LS" && isfield(receiver,'LSInterpolation')
 r.DownlinkLayout=v3LSLayouts(r,receiver.LSInterpolation);
 receiver.PriorsRX=receiver.PriorsLS;
end
[h,v,observed]=v2EstimateRX(input,receiver,r,method);
end
