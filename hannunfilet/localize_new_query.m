%% localize_new_query.m
% Use this to locate a new measurement from real data.

load('fingerprint_db.mat');    % loads F, P, col_labels, mu_F, std_F
load('wknn_results.mat');      % loads pca_coeff, n_comp, mu_F, std_F, K, etc.

% --- Build your query feature vector the same way as the database ---
% q_raw must be [1 x 333]: same feature order as F
% (rss, path_loss, delay, aoa_az, aoa_el, cov_diag_real x16, cov_diag_imag x16) x 9 TXs
q_raw = [ ... ];   % fill from your measurement

% --- Apply same normalization ---
q_norm = (q_raw - mu_F) ./ std_F;

% --- Project into PCA space ---
if use_pca
    q_proj = q_norm * pca_coeff;   % 1 x n_comp
else
    q_proj = q_norm;
end

% --- Rebuild normalized+PCA-reduced database ---
F_norm    = (F - mu_F) ./ std_F;
F_reduced = F_norm * pca_coeff;

% --- WKNN match ---
[pos_est, nn_idx, nn_dist] = wknn_match(q_proj, F_reduced, P, K, ...
                                         dist_type, zeros(1,n_comp), ones(1,n_comp));

fprintf('Estimated position: x=%.2f, y=%.2f, z=%.2f\n', pos_est(1), pos_est(2), pos_est(3));