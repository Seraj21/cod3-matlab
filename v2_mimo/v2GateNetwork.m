function net=v2GateNetwork(R,U)
width=64;
graph=layerGraph([imageInputLayer([6 4 8*R*U],'Normalization','none','Name','input')
 convolution2dLayer(1,width,'Name','embed')
 leakyReluLayer(.1,'Name','start')]);
previous='start';
for block=1:4
 prefix=sprintf('gate%d_',block);
 branch=[groupedConvolution2dLayer(3,1,width,'Padding','same','Name',[prefix 'depth'])
  convolution2dLayer(1,width,'Name',[prefix 'mix'])
  leakyReluLayer(.1,'Name',[prefix 'activation'])
  additionLayer(2,'Name',[prefix 'sum'])];
 graph=addLayers(graph,branch);graph=connectLayers(graph,previous,[prefix 'depth']);
 graph=connectLayers(graph,previous,[prefix 'sum/in2']);previous=[prefix 'sum'];
end
last=[globalAveragePooling2dLayer('Name','pool')
 fullyConnectedLayer(5,'WeightsInitializer','zeros','BiasInitializer','zeros','Name','weights')
 softmaxLayer('Name','probabilities')];
graph=addLayers(graph,last);graph=connectLayers(graph,previous,'pool');net=dlnetwork(graph);
end
