function cfg=v2ConfigFromRow(row,numRB)
cfg=v2Config(row.Antennas,numRB);
for name=["DelaySpread" "Speed" "VerticalCorrelation" "HorizontalCorrelation" "UECorrelation" "SoundingSNR"]
 cfg.(name)=row.(name);
end
end
