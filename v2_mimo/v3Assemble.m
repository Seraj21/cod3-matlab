function model=v3Assemble(baseFile,rxRoot)
loaded=load(baseFile,'model');model=loaded.model;assert(model.Version==5);
for scheme=["MR" "ZF" "PZF"]
 files=dir(fullfile(rxRoot,'**','v3_rx_'+lower(scheme)+'.mat'));assert(numel(files)==1);
 item=load(fullfile(files(1).folder,files(1).name),'receiver');receiver=item.receiver;
 assert(receiver.Version==6 && receiver.PilotMode=="DMRS+PTRS" && receiver.TrainingCodeRate==.125);
 assert(isequal(receiver.Signature,model.Signature));
 model.Receivers.(scheme)=receiver;model.Selection.(scheme)=[0 1];
end
model.Version=6;model.Signature.PilotMode='DMRS+PTRS';
model.Signature.PZFMode='robust SLNR, quantized desired-phase anchor';model.Signature.TargetCodeRate=.125;
end
