%% mlp_localization.m
load('fingerprint_db.mat');
% Filter to clean (non-imputed) RX only
load('fingerprint_db.mat');
filename = 'ray_data.h5';
reached_mask = true(size(F,1), 1);
for tx_idx = 1:9
    tx_group = sprintf('/tx_%d', tx_idx);
    rx_idx_vec = double(h5read(filename, [tx_group '/rx_index']));
    reached_by_tx = false(size(F,1), 1);
    reached_by_tx(unique(rx_idx_vec)) = true;
    reached_mask = reached_mask & reached_by_tx;
end
F = F(reached_mask, :);
P = P(reached_mask, :);
fprintf('Using %d clean RX points\n', size(F,1));
rng(42);

mu_F = mean(F,1); std_F = std(F,0,1); std_F(std_F==0)=1;
F_norm = (F - mu_F) ./ std_F;

mu_P = mean(P,1); std_P = std(P,0,1); std_P(std_P==0)=1;
P_norm = (P - mu_P) ./ std_P;

num_rx = size(F,1);
idx = randperm(num_rx);
n_tr = round(0.70*num_rx); n_va = round(0.15*num_rx);
idx_tr = idx(1:n_tr); idx_va = idx(n_tr+1:n_tr+n_va); idx_te = idx(n_tr+n_va+1:end);

X_tr = F_norm(idx_tr,:); Y_tr = P_norm(idx_tr,:);
X_va = F_norm(idx_va,:); Y_va = P_norm(idx_va,:);
X_te = F_norm(idx_te,:); Y_te = P_norm(idx_te,:);

layers = [
    featureInputLayer(size(F_norm,2), 'Normalization','none')
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.1)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.1)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 2000, 'MiniBatchSize', 128, ...
    'InitialLearnRate', 5e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 400, ...
    'L2Regularization', 1e-3, ...
    'ValidationData', {X_va, Y_va}, ...
    'ValidationFrequency', 20, ...
    'OutputNetwork', 'best-validation-loss', ...
    'ValidationPatience', 30, ...
    'Plots', 'training-progress', 'Verbose', true, 'VerboseFrequency', 50);

net = trainNetwork(X_tr, Y_tr, layers, options);

Y_pred     = predict(net, X_te);
Y_pred_raw = Y_pred .* std_P + mu_P;
Y_te_raw   = Y_te   .* std_P + mu_P;
errors     = sqrt(sum((Y_pred_raw - Y_te_raw).^2, 2));

fprintf('\n--- MLP Results (Test N=%d) ---\n', numel(idx_te));
fprintf('  MAE : %.2f m\n', mean(errors));
fprintf('  RMSE: %.2f m\n', sqrt(mean(errors.^2)));
fprintf('  P50 : %.2f m\n', median(errors));
fprintf('  P90 : %.2f m\n', prctile(errors,90));