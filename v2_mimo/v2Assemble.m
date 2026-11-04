function model=v2Assemble(txFile,rxRoot)
loaded=load(txFile,'model');model=loaded.model;
model.Selection=struct;model.Receivers=struct;
for scheme=["MR" "ZF" "PZF"]
 files=dir(fullfile(rxRoot,'**','v2_rx_'+lower(scheme)+'.mat'));
 keep=~contains(string({files.folder}),'smoke');files=files(keep);assert(numel(files)==1);
 data=load(fullfile(files(1).folder,files(1).name),'receiver');
 assert(data.receiver.Version==5 && isequal(data.receiver.Signature,model.Signature));
 model.Receivers.(scheme)=data.receiver;model.Selection.(scheme)=[1 1];
end
end
