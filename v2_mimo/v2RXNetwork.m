function net=v2RXNetwork(inputChannels,coefficients)
width=48;
graph=layerGraph([imageInputLayer([6 12 inputChannels],'Normalization','none','Name','input')
 convolution2dLayer(1,width,'Name','embed')
 leakyReluLayer(.1,'Name','start')]);previous='start';
for block=1:6
 prefix=sprintf('rx%d_',block);dilation=[1 2 4 1 2 4];
 branch=[groupedConvolution2dLayer(3,1,width,'Padding','same', ...
   'DilationFactor',[1 dilation(block)],'Name',[prefix 'depth'])
  convolution2dLayer(1,width,'Name',[prefix 'mix'])
  leakyReluLayer(.1,'Name',[prefix 'activation'])
  additionLayer(2,'Name',[prefix 'sum'])];
 graph=addLayers(graph,branch);graph=connectLayers(graph,previous,[prefix 'depth']);
 graph=connectLayers(graph,previous,[prefix 'sum/in2']);previous=[prefix 'sum'];
end
graph=addLayers(graph,convolution2dLayer(1,12*coefficients,'WeightsInitializer','zeros', ...
 'BiasInitializer','zeros','Name','mean_and_log_variance'));
graph=connectLayers(graph,previous,'mean_and_log_variance');net=dlnetwork(graph);
end
