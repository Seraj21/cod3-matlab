function v2_train_rx(scheme)
addpath('mu_mimo','v2_mimo');
v2TrainRX(string(scheme),'v2_tx_training/v2_tx.mat','v2_rx_'+lower(string(scheme)),40);
end
