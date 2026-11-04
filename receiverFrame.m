function reference = receiverFrame(channel,phaseStep,symbolCenters,frequency,offset,channelOffset,sampleRate)
% Express a channel target in the receiver's timing and CFO-compensated frame.
rotation = exp(-1i*2*pi*frequency*symbolCenters/sampleRate);
reference = channel.*exp(1i*phaseStep*(offset-channelOffset)) ...
    .*reshape(rotation,1,numel(symbolCenters),1);
end
