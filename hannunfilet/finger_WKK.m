%% finger_WKK.m
% Weighted KNN fingerprint matching with K sweep.
% Requires fingerprint_db.mat from build_fingerprint_db.m

load('fingerprint_db.mat');   % loads F, P, col_labels

%% --- Parameters ---
K_values   = [1, 3, 5, 7, 9, 11, 15];
dist_type  = 'euclidean';
normalize  = true;

%% --- Normalize ---
if normalize
    mu_F  = mean(F, 1);
    std_F = std(F, 0, 1);
    std_F(std_F == 0) = 1;
    F_norm = (F - mu_F) ./ std_F;
else
    F_norm = F;
    mu_F   = zeros(1, size(F,2));
    std_F  = ones(1, size(F,2));
end

num_rx   = size(F, 1);
num_feat = size(F_norm, 2);

%% --- WKNN function ---
function [pos_est, nn_idx, nn_dist] = wknn_match(query_vec, F_db, P_db, k, dtype, mu, sigma)
    q = (query_vec - mu) ./ sigma;

    switch dtype
        case 'euclidean'
            diffs = F_db - q;
            dists = sqrt(sum(diffs.^2, 2));
        case 'cosine'
            q_n  = q / (norm(q) + eps);
            F_n  = F_db ./ (sqrt(sum(F_db.^2, 2)) + eps);
            dists = 1 - F_n * q_n';
    end

    [sorted_d, sorted_idx] = sort(dists, 'ascend');
    nn_idx  = sorted_idx(1:k);
    nn_dist = sorted_d(1:k);

    w = 1 ./ (nn_dist + 1e-10);
    w = w / sum(w);
    pos_est = w' * P_db(nn_idx, :);
end

%% --- K Sweep ---
sweep_results = zeros(length(K_values), 5);
best_errors = [];
best_P_est  = [];

fprintf('Running K sweep (LOO-CV)...\n\n');

for k_idx = 1:length(K_values)
    K = K_values(k_idx);

    P_est  = zeros(num_rx, size(P,2));
    errors = zeros(num_rx, 1);

    for i = 1:num_rx
        idx_db = [1:i-1, i+1:num_rx];
        F_db_i = F_norm(idx_db, :);
        P_db_i = P(idx_db, :);
        q_norm = (F(i,:) - mu_F) ./ std_F;

        [P_est(i,:), ~, ~] = wknn_match(q_norm, F_db_i, P_db_i, K, dist_type, ...
                                          zeros(1, num_feat), ones(1, num_feat));
        errors(i) = norm(P_est(i,:) - P(i,:));
    end

    mae  = mean(errors);
    rmse = sqrt(mean(errors.^2));
    p50  = median(errors);
    p90  = prctile(errors, 90);
    p95  = prctile(errors, 95);

    sweep_results(k_idx,:) = [mae, rmse, p50, p90, p95];
    fprintf('K=%2d | MAE=%.2fm  RMSE=%.2fm  P50=%.2fm  P90=%.2fm  P95=%.2fm\n', ...
            K, mae, rmse, p50, p90, p95);

    if k_idx == 1 || mae < min(sweep_results(1:k_idx-1, 1))
        best_errors = errors;
        best_P_est  = P_est;
        best_K      = K;
    end
end

%% --- Best K summary ---
[~, best_idx] = min(sweep_results(:,1));
fprintf('\nBest K = %d  (MAE=%.2fm)\n', K_values(best_idx), sweep_results(best_idx,1));

mae  = sweep_results(best_idx,1);
rmse = sweep_results(best_idx,2);
p50  = sweep_results(best_idx,3);
p90  = sweep_results(best_idx,4);
p95  = sweep_results(best_idx,5);

fprintf('\n--- WKNN Localization Results (LOO-CV, Best K=%d) ---\n', best_K);
fprintf('  MAE   : %.2f m\n',  mae);
fprintf('  RMSE  : %.2f m\n',  rmse);
fprintf('  50th percentile: %.2f m\n', p50);
fprintf('  90th percentile: %.2f m\n', p90);
fprintf('  95th percentile: %.2f m\n', p95);

%% --- CDF plot ---
figure;
sorted_err = sort(best_errors);
cdf_vals   = (1:num_rx)' / num_rx;
plot(sorted_err, cdf_vals, 'b-', 'LineWidth', 2);
grid on;
xlabel('Localization Error (m)');
ylabel('CDF');
title(sprintf('WKNN Localization Error CDF  (K=%d)', best_K));
xline(mae, 'r--', sprintf('MAE=%.2fm', mae),  'LabelVerticalAlignment','bottom');
xline(p50, 'g--', sprintf('P50=%.2fm', p50),  'LabelVerticalAlignment','bottom');
xline(p90, 'm--', sprintf('P90=%.2fm', p90),  'LabelVerticalAlignment','bottom');

%% --- Save ---
K = best_K;
P_est = best_P_est;
errors = best_errors;
save('wknn_results.mat','best_errors', 'P', 'best_K', 'P_est', 'errors', 'mae', 'rmse', 'p50', 'p90', 'p95', ...
     'K', 'normalize', 'dist_type', 'mu_F', 'std_F', 'sweep_results', 'K_values');
fprintf('\nResults saved to wknn_results.mat\n');