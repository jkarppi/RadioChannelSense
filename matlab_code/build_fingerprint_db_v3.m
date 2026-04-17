%% build_fingerprint_db.m
filename = 'ray_data.h5';

%% --- Configuration ---
use_features = struct(...
    'rss',        1, ...
    'path_loss',  1, ...
    'delay',      1, ...
    'aoa_az',     1, ...
    'aoa_el',     1, ...
    'aod_az',     0, ...
    'aod_el',     0, ...
    'doppler',    0, ...
    'gain_real',  0, ...
    'gain_imag',  0  ...
);

use_cov_eig  = true;    % eigenvalue features from covariance matrix
num_eig_keep = 3;       % number of eigenvalues to keep per TX
num_ant      = 16;

%% --- Read global data ---
rx_positions = h5read(filename, '/rx_positions');   % [num_rx x 3]
tx_positions = h5read(filename, '/tx_positions');
tx_names     = h5read(filename, '/tx_names');

num_rx = size(rx_positions, 1);
num_tx = size(tx_positions, 1);

feature_names   = fieldnames(use_features);
active_features = feature_names(cellfun(@(f) use_features.(f)==1, feature_names));
num_feat_per_tx = numel(active_features);
eig_feats_per_tx = num_eig_keep * use_cov_eig;
total_feats_per_tx = num_feat_per_tx + eig_feats_per_tx;

fprintf('Building fingerprint database:\n');
fprintf('  RX locations  : %d\n', num_rx);
fprintf('  TX nodes      : %d\n', num_tx);
fprintf('  Scalar/TX     : %d  -> %s\n', num_feat_per_tx, strjoin(active_features, ', '));
fprintf('  Eigenvals/TX  : %d\n', eig_feats_per_tx);
fprintf('  Total features: %d\n\n', num_tx * total_feats_per_tx);

%% --- Allocate ---
F = zeros(num_rx, num_tx * total_feats_per_tx);
P = rx_positions;   % [num_rx x 3]

%% --- Fill fingerprint matrix ---
for tx_idx = 1:num_tx
    tx_group = sprintf('/tx_%d', tx_idx);

    % --- Read all ray-level data ---
    rx_idx_vec = double(h5read(filename, [tx_group '/rx_index']));  % [total_rays x 1]
    pl_vec     = double(h5read(filename, [tx_group '/path_loss']));

    % Read scalar ray features (except RSS which is already per-RX)
    ray_feats = struct();
    for f_idx = 1:num_feat_per_tx
        feat = active_features{f_idx};
        if ~strcmp(feat, 'rss')
            ray_feats.(feat) = double(h5read(filename, [tx_group '/' feat]));
        end
    end

    % RSS and covariance are already per-unique-RX
    rss_per_rx   = double(h5read(filename, [tx_group '/rss']));
    unique_rx_ids = unique(rx_idx_vec);   % sorted global 1-based RX indices
    num_reached   = numel(unique_rx_ids);

    % --- Build scalar feature block (best path per RX) ---
    tx_scalar = NaN(num_rx, num_feat_per_tx);

    for r = 1:num_reached
        rx_id     = unique_rx_ids(r);
        path_mask = rx_idx_vec == rx_id;
        pl_this   = pl_vec(path_mask);
        [~, best_local] = min(pl_this);         % strongest path = min path_loss
        all_idx   = find(path_mask);
        best_idx  = all_idx(best_local);

        fvec = zeros(1, num_feat_per_tx);
        for f_idx = 1:num_feat_per_tx
            feat = active_features{f_idx};
            if strcmp(feat, 'rss')
                fvec(f_idx) = rss_per_rx(r);
            else
                fvec(f_idx) = ray_feats.(feat)(best_idx);
            end
        end
        tx_scalar(rx_id, :) = fvec;
    end

    % --- Impute missing RX (not reached by this TX) ---
    missing = isnan(tx_scalar(:,1));
    if any(missing)
        for f_idx = 1:num_feat_per_tx
            feat = active_features{f_idx};
            valid = tx_scalar(~missing, f_idx);
            if strcmp(feat, 'rss')
                imp = min(valid) - 10;      % 10dB below minimum
            elseif strcmp(feat, 'path_loss')
                imp = max(valid) + 10;      % 10dB above maximum
            else
                imp = mean(valid);          % mean for angles/delay
            end
            tx_scalar(missing, f_idx) = imp;
        end
        fprintf('  %s: %d missing RX imputed\n', tx_group, sum(missing));
    end

    % --- Eigenvalue feature block ---
    if use_cov_eig
        cov_r     = double(h5read(filename, [tx_group '/covariance_real']));
        cov_i     = double(h5read(filename, [tx_group '/covariance_imag']));
        tx_eig    = zeros(num_rx, num_eig_keep);   % missing RX → zero eigenvalues

        for r = 1:num_reached
            rx_id = unique_rx_ids(r);
            R  = squeeze(cov_r(r,:,:)) + 1j * squeeze(cov_i(r,:,:));
            ev = sort(real(eig(R)), 'descend');
            tx_eig(rx_id, :) = ev(1:num_eig_keep)';
        end
    else
        tx_eig = [];
    end

    % --- Insert into F ---
    col_start = (tx_idx-1) * total_feats_per_tx + 1;
    col_end   = col_start + total_feats_per_tx - 1;
    F(:, col_start:col_end) = [tx_scalar, tx_eig];

    fprintf('  Processed /tx_%d (%d/%d RX reached)\n', tx_idx, num_reached, num_rx);
end

% Remove z-coordinate from P (all same height)
P = rx_positions(:, 1:2);   % keep only x,y for 2D localization

fprintf('\nFingerprint database saved to fingerprint_db.mat\n');
fprintf('  F: [%d x %d]  (observations x features)\n', size(F,1), size(F,2));
fprintf('  P: [%d x %d]  (x, y positions)\n', size(P,1), size(P,2));

col_labels = {};
for tx_idx = 1:num_tx
    for f = active_features'
        col_labels{end+1} = sprintf('tx%d_%s', tx_idx, f{1});
    end
    if use_cov_eig
        for e = 1:num_eig_keep
            col_labels{end+1} = sprintf('tx%d_eig%d', tx_idx, e);
        end
    end
end

%% --- Remove RX with any imputed (missing) data ---
% Build mask: keep only RX reached by ALL TXs
% Binary reached flags [3186 x 9]
reached_flags = zeros(num_rx, num_tx);
for tx_idx = 1:num_tx
    tx_group = sprintf('/tx_%d', tx_idx);
    rx_idx_vec = double(h5read(filename, [tx_group '/rx_index']));
    reached_flags(unique(rx_idx_vec), tx_idx) = 1;
end
F = [F, reached_flags];   % append 9 binary features → [3186 x 81]
fprintf('  Appended %d reached-flag features\n', num_tx);

% NOTE: Do NOT filter F/P here — keep all 3186 for MLP
% (MLP uses reached flags to handle blocked paths)

fprintf('\nFingerprint database saved to fingerprint_db.mat\n');
fprintf('  F: [%d x %d] ...\n', size(F,1), size(F,2));

save('fingerprint_db.mat', 'F', 'P', 'col_labels');