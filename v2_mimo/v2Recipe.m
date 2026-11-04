function T=v2Recipe(trainingFrames,validationFrames)
stream=RandStream('mt19937ar','Seed',710011);
rows=cell(trainingFrames+validationFrames,15);at=0;
for split=1:2
 count=trainingFrames;if split==2,count=validationFrames;end
 schemes=repmat(1:3,1,ceil(count/3));schemes=schemes(1:count);schemes=schemes(randperm(stream,count));
 sources=repmat(1:4,1,ceil(count/4));sources=sources(1:count);sources=sources(randperm(stream,count));
 for frame=1:count
  cfg=v2Config(32+32*mod(floor((frame-1)/2),2),12);
  hard=mod(frame,2)==0;
  if ~hard
   speeds=[0 30 60 90 120];cfg.Speed=speeds(1+mod(floor((frame-1)/2),5));
   cfg.DelaySpread=10^(-7+rand(stream));
   if split==2
    delays=[.1 .3 1]*1e-6;cfg.DelaySpread=delays(1+mod(floor((frame-1)/10),3));
   end
   cfg.VerticalCorrelation=.4+.5*rand(stream);cfg.HorizontalCorrelation=.1+.4*rand(stream);
   cfg.UECorrelation=.5*rand(stream);cfg.SoundingSNR=5+20*rand(stream);
  end
  seed=7000000+frame;if split==2,seed=8000000+frame;end
  at=at+1;
  rows(at,:)={split==1,frame,seed,cfg.Antennas,cfg.DelaySpread,cfg.Speed, ...
   cfg.VerticalCorrelation,cfg.HorizontalCorrelation,cfg.UECorrelation,cfg.SoundingSNR, ...
   schemes(frame),sources(frame),20*rand(stream),8+12*rand(stream),hard};
 end
end
T=cell2table(rows,'VariableNames',{'Training','Frame','Seed','Antennas','DelaySpread','Speed', ...
 'VerticalCorrelation','HorizontalCorrelation','UECorrelation','SoundingSNR', ...
 'Scheme','CSITSource','DataSNR1','DataSNR2','HardCase'});
end
