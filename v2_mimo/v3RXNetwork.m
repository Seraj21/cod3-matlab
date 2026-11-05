function net=v3RXNetwork(inputChannels,coefficients,warm)
% Compact residual channel estimator with two squeeze/excitation gates.
% Split-attention literature motivates channel selection; not a ResNeSt-50 copy.
width=48;graph=layerGraph([imageInputLayer([6 12 inputChannels],'Normalization','none','Name','input')
 convolution2dLayer(1,width,'Name','embed');leakyReluLayer(.1,'Name','start')]);previous='start';
for block=1:6
 prefix=sprintf('rx%d_',block);dilation=[1 2 4 1 2 4];
 branch=[groupedConvolution2dLayer(3,1,width,'Padding','same', ...
   'DilationFactor',[1 dilation(block)],'Name',[prefix 'depth'])
  convolution2dLayer(1,width,'Name',[prefix 'mix'])
  leakyReluLayer(.1,'Name',[prefix 'activation'])];
 graph=addLayers(graph,branch);graph=connectLayers(graph,previous,[prefix 'depth']);out=[prefix 'activation'];
 if any(block==[3 6])
  attention=[globalAveragePooling2dLayer('Name',[prefix 'pool'])
   convolution2dLayer(1,12,'Name',[prefix 'attention_reduce'])
   reluLayer('Name',[prefix 'attention_relu'])
   convolution2dLayer(1,width,'WeightsInitializer','zeros','BiasInitializer','zeros','Name',[prefix 'attention_expand'])
   sigmoidLayer('Name',[prefix 'attention_sigmoid'])];
  graph=addLayers(graph,attention);graph=connectLayers(graph,out,[prefix 'pool']);
  gate=functionLayer(@(x,g)x.*(2*g),'NumInputs',2,'Formattable',true,'Name',[prefix 'gate']);
  graph=addLayers(graph,gate);graph=connectLayers(graph,out,[prefix 'gate/in1']);
  graph=connectLayers(graph,[prefix 'attention_sigmoid'],[prefix 'gate/in2']);out=[prefix 'gate'];
 end
 graph=addLayers(graph,additionLayer(2,'Name',[prefix 'sum']));
 graph=connectLayers(graph,out,[prefix 'sum/in1']);graph=connectLayers(graph,previous,[prefix 'sum/in2']);
 previous=[prefix 'sum'];
end
graph=addLayers(graph,convolution2dLayer(1,12*coefficients,'WeightsInitializer','zeros', ...
 'BiasInitializer','zeros','Name','mean_and_log_variance'));
graph=connectLayers(graph,previous,'mean_and_log_variance');net=dlnetwork(graph);
if nargin>=3
 values=net.Learnables;
 for i=1:height(values)
  if string(values.Layer(i))=="mean_and_log_variance",continue;end
  j=find(string(warm.Learnables.Layer)==string(values.Layer(i)) & ...
      string(warm.Learnables.Parameter)==string(values.Parameter(i)));
  if ~isempty(j),values.Value{i}=warm.Learnables.Value{j};end
 end
 net=dlupdate(@(~,value)value,net,values);
end
end
