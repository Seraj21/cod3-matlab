function net = muCNNArchitecture(inputChannels,coefficients)
% Joint spatial/frequency/time residual network at an unshuffled frequency grid.
width = 32;
initial = [imageInputLayer([6 12 inputChannels],'Normalization','none','Name','input')
    convolution2dLayer(1,width,'Name','embed')
    leakyReluLayer(.1,'Name','activation0')];
graph = layerGraph(initial);
previous = 'activation0';
dilations = [1 2 4 1];
for b = 1:numel(dilations)
    prefix = sprintf('block%d_',b);
    branch = [groupedConvolution2dLayer([3 3],1,width,'Padding','same', ...
        'DilationFactor',[1 dilations(b)],'Name',[prefix 'depthwise'])
        convolution2dLayer(1,width,'Name',[prefix 'mix'])
        leakyReluLayer(.1,'Name',[prefix 'activation'])
        additionLayer(2,'Name',[prefix 'sum'])];
    graph = addLayers(graph,branch);
    graph = connectLayers(graph,previous,[prefix 'depthwise']);
    graph = connectLayers(graph,previous,[prefix 'sum/in2']);
    previous = [prefix 'sum'];
end
output = convolution2dLayer(1,8*coefficients,'WeightsInitializer','zeros', ...
    'BiasInitializer','zeros','Name','correction');
graph = addLayers(graph,output);
graph = connectLayers(graph,previous,'correction');
net = dlnetwork(graph);
end
