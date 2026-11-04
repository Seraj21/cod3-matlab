function v2_prepare_tx(trainingFrames,validationFrames,epochs)
if nargin<1,trainingFrames=1920;end
if nargin<2,validationFrames=240;end
if nargin<3,epochs=50;end
addpath('mu_mimo','v2_mimo');
v2PrepareTX('v2_tx_training',trainingFrames,validationFrames,epochs);
end
