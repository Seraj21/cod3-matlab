clear; clc; close all;

fc = 28e9;
c_light = 3e8;
User_Speed_kmh = 30;
User_Speed_ms = User_Speed_kmh * (1000/3600);
fd_max = (User_Speed_ms / c_light) * fc;

SCS_kHz = 60;
SCS = SCS_kHz * 1e3;
BW_MHz = 50;
N_RB = 66;
N_active = N_RB * 12;

N_fft = 1024;
Fs = N_fft * SCS;
Ts = 1/Fs;
N_cp = round(N_fft * 0.07);
T_sym_total = (N_fft + N_cp) * Ts;
System_BW = BW_MHz * 1e6;

tdl_c_pdp_db = [0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0, -8.9, -9.0];
tdl_c_delays_ns = [0, 30, 70, 90, 110, 190, 410, 2000, 2300, 3400, 4800, 5200];
tdl_c_pdp_lin = 10.^(tdl_c_pdp_db/10);
pdb = tdl_c_pdp_lin / sum(tdl_c_pdp_lin);
path_delays_samples = round((tdl_c_delays_ns * 1e-9) / Ts);
L_paths = length(pdb);

configs_matrix = [
    1,  1,  1, 1;
    1,  4,  1, 4;
    4,  1,  4, 1;
    4,  4,  4, 4;
    16, 16, 4, 4;
    32, 32, 8, 8
];
config_names = {'SISO (1x1)', 'SIMO (1x4)', 'MISO (4x1)', 'MIMO (4x4)', 'mMIMO (16x16 HBF)', 'mMIMO (32x32 HBF)'};

M = 256;
bps = log2(M);
snr_db_range = 0:5:30;
num_frames = 20;
symbols_per_frame = 14;
N_stream = 1;

fprintf('======================================================\n');
fprintf('      5G mmWave Hybrid Beamforming Simulation         \n');
fprintf('======================================================\n');
fprintf('Carrier Freq: %.1f GHz | Bandwidth: %d MHz | Subcarriers: %d \n', fc/1e9, BW_MHz ,N_active );
fprintf('Channel Model: 3GPP TDL-C \n');
fprintf('Beamforming: Hybrid (Analog + Digital)\n');
fprintf('======================================================\n\n');

ber_results = zeros(size(configs_matrix, 1), length(snr_db_range));
thr_results = zeros(size(configs_matrix, 1), length(snr_db_range));
se_results  = zeros(size(configs_matrix, 1), length(snr_db_range));

for c = 1:size(configs_matrix, 1)
    Nt = configs_matrix(c, 1);
    Nr = configs_matrix(c, 2);
    N_rf_tx = configs_matrix(c, 3);
    N_rf_rx = configs_matrix(c, 4);

    fprintf('Running: %s (RF Chains: Tx=%d, Rx=%d)...\n', config_names{c}, N_rf_tx, N_rf_rx);

    for s_idx = 1:length(snr_db_range)
        snr_db = snr_db_range(s_idx);
        snr_lin = 10^(snr_db/10);
        noise_var = 1 / snr_lin;

        total_bits = 0;
        total_err_bits = 0;

        for f = 1:num_frames
            H_all = zeros(Nr, Nt, N_active);
            for l = 1:L_paths
                h_tap = (randn(Nr, Nt) + 1j*randn(Nr, Nt)) / sqrt(2);
                for k = 1:N_active
                    delay_phase = exp(-1j * 2 * pi * (k-1) * path_delays_samples(l) / N_fft);
                    H_all(:, :, k) = H_all(:, :, k) + sqrt(pdb(l)) * h_tap * delay_phase;
                end
            end

            R_tx = zeros(Nt, Nt);
            R_rx = zeros(Nr, Nr);
            for k = 1:N_active
                H = H_all(:, :, k);
                R_tx = R_tx + H' * H;
                R_rx = R_rx + H * H';
            end
            R_tx = R_tx / N_active;
            R_rx = R_rx / N_active;

            [V_tx, ~] = eig(R_tx);
            F_RF = exp(1j * angle(V_tx(:, end-N_rf_tx+1:end)));

            [V_rx, ~] = eig(R_rx);
            W_RF = exp(1j * angle(V_rx(:, end-N_rf_rx+1:end)));

            nbits = N_active * symbols_per_frame * N_stream * bps;
            tx_bits = randi([0 1], nbits, 1);
            tx_syms = qammod(tx_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
            tx_grid = reshape(tx_syms, N_stream, N_active, symbols_per_frame);
            rx_grid = zeros(N_stream, N_active, symbols_per_frame);

            for sym = 1:symbols_per_frame
                for k = 1:N_active
                    H = H_all(:, :, k);

                    H_eff = W_RF' * H * F_RF;

                    [U_eff, S_eff, V_eff] = svd(H_eff);
                    F_BB = V_eff(:, 1:N_stream);

                    F_total = F_RF * F_BB;
                    scale_factor = sqrt(N_stream) / norm(F_total, 'fro');
                    F_BB = F_BB * scale_factor;

                    x = tx_grid(:, k, sym);
                    x_transmitted = F_RF * (F_BB * x);

                    noise = sqrt(noise_var/2) * (randn(Nr, 1) + 1j*randn(Nr, 1));
                    y = H * x_transmitted + noise;

                    y_rf = W_RF' * y;

                    noise_cov = (W_RF' * W_RF) * noise_var;

                    H_total = W_RF' * H * F_RF * F_BB;

                    W_BB = H_total' * pinv(H_total * H_total' + noise_cov);

                    x_est = W_BB * y_rf;

                    gains = diag(W_BB * H_total);
                    x_est = x_est ./ gains;

                    rx_grid(:, k, sym) = x_est;
                end
            end

            rx_syms_flat = rx_grid(:);
            rx_bits = qamdemod(rx_syms_flat, M, 'OutputType', 'bit', 'UnitAveragePower', true);
            total_err_bits = total_err_bits + sum(abs(tx_bits - rx_bits));
            total_bits = total_bits + nbits;
        end

        ber_results(c, s_idx) = total_err_bits / total_bits;

        correct_bits = total_bits - total_err_bits;
        total_duration = num_frames * symbols_per_frame * T_sym_total;
        thr_results(c, s_idx) = correct_bits / total_duration;
        se_results(c, s_idx) = thr_results(c, s_idx) / System_BW;

        if mod(s_idx, 2) == 0 || s_idx == 1
             fprintf('   SNR %2d dB | BER: %.5f | SE: %.2f\n', snr_db, ber_results(c, s_idx), se_results(c, s_idx));
        end
    end
    fprintf('\n');
end

all_markers = {'-o', '-s', '-^', '-d', '-h', '-p'};
all_colors = {'k', 'b', 'r', 'g', 'm', [0.85, 0.32, 0.09]};
linewidths = [1.5, 1.5, 1.5, 2, 2.5, 3];

figure('Name', 'Hybrid Beamforming BER', 'Color', 'w');
for c = 1:size(configs_matrix, 1)
    semilogy(snr_db_range, ber_results(c, :), all_markers{c}, ...
        'Color', all_colors{c}, 'LineWidth', linewidths(c), 'MarkerSize', 8);
    hold on;
end
grid on;
title('BER Performance: Hybrid Beamforming (TDL-C Channel)', 'FontSize', 12);
xlabel('SNR (dB)', 'FontWeight', 'bold'); ylabel('BER', 'FontWeight', 'bold');
legend(config_names, 'Location', 'SouthWest');
ylim([1e-6 1]);

figure('Name', 'Hybrid Beamforming SE', 'Color', 'w');
for c = 1:size(configs_matrix, 1)
    plot(snr_db_range, se_results(c, :), all_markers{c}, ...
        'Color', all_colors{c}, 'LineWidth', linewidths(c), 'MarkerSize', 8);
    hold on;
end
grid on;
title('Spectral Efficiency', 'FontSize', 12);
xlabel('SNR (dB)', 'FontWeight', 'bold'); ylabel('SE (bps/Hz)', 'FontWeight', 'bold');
legend(config_names, 'Location', 'NorthWest');
