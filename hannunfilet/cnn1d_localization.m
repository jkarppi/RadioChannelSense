%% cnn1d_localization.m
% 1D-CNN for 3D indoor localization
% Inspired by: "Three-Dimensional Indoor Positioning with 802.11az
%  Fingerprinting and Deep Learning"
%
% 1D conv implemented as convolution2dLayer([k 1], filters):
%   filters scan the feature vector (height), width is fixed at 1.
% Input shape: [num_feats x 1 x 1 x N] = [45 x 1 x 1 x 810]

load('fingerprint_db.mat');   % F: 810x45, P: 810x3

%% --- Parameters ---
rng(42);
num_rx    = size(F, 1);   % 810
num_feats = size(F, 2);   % 45

%% --- Normalize (recompute — do NOT load from WKNN results) ---
mu_F  = mean(F, 1);
std_F = std(F, 0, 1);
std_F(std_F == 0) = 1;
F_norm = (F - mu_F) ./ std_F;   % 810 x 45

%% --- Normalize targets (positions) ---
mu_P  = mean(P, 1);
std_P = std(P, 0, 1);
std_P(std_P == 0) = 1;
P_norm = (P - mu_P) ./ std_P;   % zero-mean, unit-variance coordinates

Y = P_norm;   % use normalized targets for training

%% --- Reshape to [H x W x C x N] for 1D-CNN ---
% H = num_feats (scanned by conv), W = 1, C = 1 (single channel), N = samples
%X = reshape(F_norm', [num_feats, 1, 1, num_rx]);   % 45 x 1 x 1 x 810
% Y = P;   % 810 x 3 already set above

num_tx_cnn  = 9;
num_f_tx    = size(F_norm, 2) / num_tx_cnn;   % features per TX, autocompute: 54/9= 6

% Reshape as 2D image: [9 x 5 x 1 x 810] — rows=TXs, cols=features per TX
% F_norm is 810 x 45, arranged as [rx, tx1_f1, tx1_f2, ..., tx9_f5]
% Reshape each row into [num_f_tx x num_tx_cnn], then permute to [H x W x C x N]
X = zeros(num_f_tx, num_tx_cnn, 1, num_rx);
for i = 1:num_rx
    X(:,:,1,i) = reshape(F_norm(i,:), [num_f_tx, num_tx_cnn]);
end


%% --- Train / Val / Test split (70 / 15 / 15) ---
idx  = randperm(num_rx);
n_tr = round(0.70 * num_rx);
n_va = round(0.15 * num_rx);

idx_tr = idx(1:n_tr);
idx_va = idx(n_tr+1:n_tr+n_va);
idx_te = idx(n_tr+n_va+1:end);

X_tr = X(:,:,:,idx_tr);   Y_tr = Y(idx_tr,:);
X_va = X(:,:,:,idx_va);   Y_va = Y(idx_va,:);
X_te = X(:,:,:,idx_te);   Y_te = Y(idx_te,:);

fprintf('Split — Train: %d | Val: %d | Test: %d\n', ...
    numel(idx_tr), numel(idx_va), numel(idx_te));

%% --- Architecture ---
layers = [
%    imageInputLayer([num_feats 1 1], 'Normalization', 'none', 'Name', 'input'),

    imageInputLayer([num_f_tx num_tx_cnn 1], 'Normalization', 'none', 'Name', 'input')
    
    % Block 1
    convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    % Block 2
    convolution2dLayer([3 3], 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    % Block 3
    convolution2dLayer([3 3], 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    % Global average pooling over spatial dims [45 x 1] -> [128] vector
    globalAveragePooling2dLayer('Name', 'gap')

    % FC head
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.3, 'Name', 'drop1')

    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.2, 'Name', 'drop2')

    fullyConnectedLayer(3, 'Name', 'fc_out')   % x, y, z
    regressionLayer('Name', 'output')
];

%% --- Training options ---
options = trainingOptions('adam', ...
    'MaxEpochs',              300, ...
    'MiniBatchSize',           64, ...
    'InitialLearnRate',       1e-3, ...
    'LearnRateSchedule',      'piecewise', ...
    'LearnRateDropFactor',      0.5, ...
    'LearnRateDropPeriod',     100, ...
    'L2Regularization',       1e-4, ...
    'GradientThreshold',        1, ...
    'ValidationData',         {X_va, Y_va}, ...
    'ValidationFrequency',     10, ...
    'Shuffle',                'every-epoch', ...
    'Plots',                  'training-progress', ...
    'Verbose',                 true, ...
    'VerboseFrequency',        50, ...
    'OutputNetwork',          'best-validation-loss');

%% --- Train ---
fprintf('\nTraining 1D-CNN...\n');
tic;
net = trainNetwork(X_tr, Y_tr, layers, options);
fprintf('Training completed in %.1f s\n', toc);

%% --- Evaluate on test set ---
Y_pred = predict(net, X_te);                           % N_te x 3
% was: errors = sqrt(sum((Y_pred - Y_te).^2, 2));

Y_pred_raw = Y_pred .* std_P + mu_P;   % back to metres
Y_te_raw   = Y_te   .* std_P + mu_P;

errors = sqrt(sum((Y_pred_raw - Y_te_raw).^2, 2));


mae  = mean(errors);
rmse = sqrt(mean(errors.^2));
p50  = median(errors);
p90  = prctile(errors, 90);
p95  = prctile(errors, 95);

fprintf('\n--- 1D-CNN Localization Results (Test Set, N=%d) ---\n', numel(idx_te));
fprintf('  MAE   : %.2f m\n',  mae);
fprintf('  RMSE  : %.2f m\n',  rmse);
fprintf('  50th percentile: %.2f m\n', p50);
fprintf('  90th percentile: %.2f m\n', p90);
fprintf('  95th percentile: %.2f m\n', p95);

%% --- CDF comparison plot ---
figure;
sorted_err = sort(errors);
n_te = numel(errors);
plot(sorted_err, (1:n_te)'/n_te, 'r-', 'LineWidth', 2);
hold on;
if exist('wknn_results.mat', 'file')
    w = load('wknn_results.mat');
    w_err = sort(w.errors);
    plot(w_err, (1:length(w_err))'/length(w_err), 'b--', 'LineWidth', 2);
    legend('1D-CNN (test set)', 'WKNN K=5 (LOO-CV)', 'Location', 'southeast');
end
grid on;
xlabel('Localization Error (m)');
ylabel('CDF');
title('1D-CNN vs WKNN — Localization Error CDF');
xline(mae, 'r:', sprintf('CNN MAE=%.2fm', mae), 'LabelVerticalAlignment', 'bottom');

%% --- Save ---
save('cnn1d_results.mat', 'net', 'Y_pred', 'Y_te', 'errors', ...
     'mae', 'rmse', 'p50', 'p90', 'p95', ...
     'mu_F', 'std_F', 'idx_tr', 'idx_va', 'idx_te', 'mu_P', 'std_P');
fprintf('Saved to cnn1d_results.mat\n');