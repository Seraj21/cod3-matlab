clear; clc; close all;

configs = [1, 1;
           1, 4;
           4, 1;
           16, 16;
           32, 32];
config_names = {'1x1 (SISO)', '1x4 (SIMO)', '4x1 (MISO)', '16x16 (MIMO)', '32x32 (mMIMO)'};
markers = {'-o', '-s', '-^', '-d', '-h'};

N_fft = 64;
N_cp = 16;
N_used = N_fft;

M = 256;
bps = log2(M);
N_stream = 1;
N_rf = 4;
snr_db_range = 0:2:30;
num_ofdm_symbols = 100;
num_monte_carlo = 10;

tdl_c_pdp_db = [0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2];
tdl_c_pdp_lin = 10.^(tdl_c_pdp_db/10);
pdb = tdl_c_pdp_lin / sum(tdl_c_pdp_lin);
L_paths = length(pdb);

ber_results = zeros(size(configs, 1), length(snr_db_range));

fprintf('Starting 3GPP OFDM Hybrid BF Simulation (%d-QAM) with MC=%d...\n', M, num_monte_carlo);

for c = 1:size(configs, 1)
    Nt = configs(c, 1);
    Nr = configs(c, 2);
    current_N_rf = min([Nt, Nr, N_rf]);

    fprintf('Simulating %s ...\n', config_names{c});

    for s_idx = 1:length(snr_db_range)
        snr_db = snr_db_range(s_idx);
        snr_lin = 10^(snr_db/10);
        noise_var = 1 / snr_lin;

        total_errors = 0;
        total_bits = 0;

        for mc = 1:num_monte_carlo

            for sym_idx = 1:num_ofdm_symbols
                bits_per_ofdm = N_used * bps;
                tx_bits = randi([0 1], bits_per_ofdm, 1);
                tx_syms_serial = qammod(tx_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
                S = reshape(tx_syms_serial, N_stream, N_used);

                h_time = zeros(Nr, Nt, L_paths);

                for l = 1:L_paths
                    h_tap = (randn(Nr, Nt) + 1j*randn(Nr, Nt)) / sqrt(2);
                    h_time(:, :, l) = sqrt(pdb(l)) * h_tap;
                end

                H_f = zeros(Nr, Nt, N_fft);
                for nr = 1:Nr
                    for nt = 1:Nt
                        H_f(nr, nt, :) = fft(squeeze(h_time(nr, nt, :)), N_fft);
                    end
                end

                R_tx_avg = zeros(Nt, Nt);
                for k = 1:N_used
                    H_k = H_f(:, :, k);
                    R_tx_avg = R_tx_avg + H_k' * H_k;
                end
                R_tx_avg = R_tx_avg / N_used;

                [V_avg, ~] = eig(R_tx_avg);
                if Nt > 1
                    F_RF = exp(1j * angle(V_avg(:, end-current_N_rf+1:end)));
                else
                    F_RF = 1;
                end

                R_rx_avg = zeros(Nr, Nr);
                for k = 1:N_used
                     H_k = H_f(:, :, k);
                     R_rx_avg = R_rx_avg + H_k * H_k';
                end
                R_rx_avg = R_rx_avg / N_used;

                [V_rx_avg, ~] = eig(R_rx_avg);
                if Nr > 1
                     W_RF = exp(1j * angle(V_rx_avg(:, end-current_N_rf+1:end)));
                else
                     W_RF = 1;
                end

                Rx_syms_grid = zeros(N_stream, N_used);

                for k = 1:N_used
                    H_k = H_f(:, :, k);
                    s_k = S(:, k);

                    H_eff_tx_k = H_k * F_RF;
                    [~, ~, V_eff_k] = svd(H_eff_tx_k);
                    F_BB_k = V_eff_k(:, 1:N_stream);

                    F_total_k = F_RF * F_BB_k;
                    scale_k = 1 / norm(F_total_k, 'fro');
                    F_BB_k = F_BB_k * scale_k;

                    x_k = F_RF * F_BB_k * s_k;

                    n_k = sqrt(noise_var/2) * (randn(Nr, 1) + 1j*randn(Nr, 1));
                    y_k = H_k * x_k + n_k;

                    H_total_eff_k = W_RF' * H_k * F_RF * F_BB_k;
                    y_rf_k = W_RF' * y_k;

                    Noise_cov_eff = (W_RF' * W_RF) * noise_var;
                    W_BB_k = H_total_eff_k' * pinv(H_total_eff_k * H_total_eff_k' + Noise_cov_eff);

                    r_est_k = W_BB_k * y_rf_k;
                    g_eff_k = W_BB_k * H_total_eff_k;
                    r_eq_k = r_est_k ./ diag(g_eff_k);

                    Rx_syms_grid(:, k) = r_eq_k;
                end

                rx_syms_serial = Rx_syms_grid(:);
                rx_bits = qamdemod(rx_syms_serial, M, 'OutputType', 'bit', 'UnitAveragePower', true);
                total_errors = total_errors + sum(abs(rx_bits - tx_bits));
                total_bits = total_bits + length(tx_bits);
            end

        end

        ber_results(c, s_idx) = total_errors / total_bits;
    end
end

figure('Name', '3GPP TDL-C Hybrid Beamforming with Monte Carlo');
linewidths = [1.5, 1.5, 1.5, 2, 2.5];
for c = 1:size(configs, 1)
    semilogy(snr_db_range, ber_results(c, :), markers{c}, 'LineWidth', linewidths(c), 'MarkerSize', 8);
    hold on;
end
grid on;
title(sprintf('Performance in 3GPP TDL-C \nModulation: %d-QAM, OFDM: %d Subcarriers', M, N_fft));
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend(config_names, 'Location', 'SouthWest');
ylim([1e-5 1]);
xlim([min(snr_db_range) max(snr_db_range)]);
