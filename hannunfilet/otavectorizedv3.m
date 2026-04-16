clear all; close all;
memory; 
tic;

% ====================== PARAMETERS ======================
fq = 3.6e9;                                      % Use 3.6 GHz to match txsite
c = physconst('LightSpeed');
lambda = c / fq;

viewer = siteviewer("Buildings","otaniemi.osm")

% Antenna (16-element ULA, λ/2 spacing)
antenna = phased.ULA('NumElements', 16, 'ElementSpacing', 0.5*lambda);


% ====================== TRANSMITTERS ======================

% Geographic origin (cartesian (0,0) maps to this lat/lon)
lat0 = 60.18687;
lon0 = 24.82948;
m_per_deg_lat = 111320;
m_per_deg_lon = 111320 * cosd(lat0);

tx_cart = [-5,70; -33,-17; -3,14; -6,-40; 55,-52.5; 61,7; 49,46; 28,-41; 84,-18];
names   = {'a','b','c','d','e','f','g','h','i'};
txs = cell(1,9);
for i = 1:9
    lat_i = lat0 + tx_cart(i,2) / m_per_deg_lat;
    lon_i = lon0 + tx_cart(i,1) / m_per_deg_lon;
    txs{i} = txsite("Name", names{i}, ...
                    "Latitude", lat_i, "Longitude", lon_i, ...
                    "AntennaHeight", 19, ...
                    "TransmitterFrequency", fq);
end

% Optional: visualize transmitters
% show(tx1); show(tx2); show(tx3); show(tx4); show(tx5); 
% show(tx6); show(tx7); show(tx8); show(tx9);

% ====================== PROPAGATION MODEL ======================
pm = propagationModel("raytracing", ...
    "Method", "image", ...
    "MaxNumReflections", 1, ...  % was 2
    "MaxRelativePathLoss", 40, ...
    "MaxAbsolutePathLoss", 160, ...
    "AngularSeparation", "medium", ...
    "SurfaceMaterial", "concrete", ...
    "BuildingsMaterial", "concrete", ...
    "TerrainMaterial", "concrete");

pm.MaxNumDiffractions = 1;

if gpuDeviceCount > 0
    pm.UseGPU = "on";
    disp('Using GPU acceleration');
end


% ====================== RECEIVER GRID ======================

rxx = -47.5:2.5:85;
rxy = -60:2.5:85;
zf  = 1.25;

[Xc, Yc] = meshgrid(rxx, rxy);
Xc = Xc(:);
Yc = Yc(:);

% Convert cartesian → geographic
rx_lats = lat0 + Yc / m_per_deg_lat;
rx_lons = lon0 + Xc / m_per_deg_lon;

all_rx = rxsite("Latitude",  rx_lats, ...
                "Longitude", rx_lons, ...
                "AntennaHeight", zf);

% all_rx= all_rx(1:200); % for testing

fprintf('Created %d receiver positions.\n', length(all_rx));


fprintf('Created %d receiver positions on a 2.5m grid.\n', length(all_rx));

% ================RAY TRACING OPTIMIZED GPU BATCHING ======================


batch_size = 50; 
ray_results = cell(1, length(txs));

for t = 1:length(txs)
    % Force GPU to clear all resident ray-tracing kernels
    gpuDevice([]);

    currentTx = txs{t};
    fprintf('Processing TX %d: %s\n', t, currentTx.Name);
    
    % Use a CELL ARRAY for accumulation, NOT a growing struct array
    % This is much more memory efficient
    num_batches = ceil(length(all_rx)/batch_size);
    tx_batches = cell(1, num_batches);
    
    batch_count = 1;
    for b = 1:batch_size:length(all_rx)
        end_idx = min(b + batch_size - 1, length(all_rx));
        current_batch = all_rx(b:end_idx);
        
        % Raytrace
        rays = raytrace(currentTx, current_batch, pm);
        
        % Extract
        
         batch_cart = [Xc(b:end_idx), Yc(b:end_idx), zf*ones(end_idx-b+1, 1)];
         tx_batches{batch_count} = extract_ray_data(rays, currentTx, current_batch, t, 0, [0 0 0], fq, b, batch_cart); 
        
        clear rays; 
        % Force MATLAB to clear internal system events and pointers
        drawnow limitrate; 
        
        batch_count = batch_count + 1;
    end
    
    % Combine all batches for this TX into one structure at the end
    % This avoids the "repeated reallocation" crash
    fprintf('  Combining %d batches for TX %d...\n', num_batches, t);
    
    combined_data = struct();
    % Get the field names from the first batch that actually has rays
    first_valid = find(~cellfun(@isempty, tx_batches), 1);
    if isempty(first_valid)
        warning('No rays found for any batch in TX %d', t);
        continue; 
    end
    
    fnames = fieldnames(tx_batches{first_valid});
    
    for fn = 1:length(fnames)
        field = fnames{fn};
        % Extract this field from every batch into a cell array
        list_of_fields = cellfun(@(s) s.(field), tx_batches, 'UniformOutput', false);
        % Stack them vertically into one long array
        combined_data.(field) = vertcat(list_of_fields{:});
    end

    ray_results{t} = combined_data;
    clear tx_batches combined_data;

end

%for tx_idx = 1:length(txs)
%    currentTx = txs{tx_idx};   
 %   fprintf('Processing transmitter %d / %d (%s)\n', ...
 %           tx_idx, length(txs), currentTx.Name);
 %   rays = raytrace(currentTx, all_rx, pm);
 %   data_s = extract_ray_data(rays, currentTx, all_rx, tx_idx, 0, [0 0 0], fq);
 %   ray_results{end+1} = data_s;
%end

fprintf('\nRay tracing completed! Total transmitter blocks: %d\n', length(ray_results));

% ====================== SAVE TO HDF5 + COVARIANCE ======================


h5_filename = 'ray_data.h5';

% Force clean deletion
if exist(h5_filename, 'file')
    try
        delete(h5_filename);
        fprintf('Existing %s deleted successfully.\n', h5_filename);
    catch ME
        warning('Could not delete %s: %s', h5_filename, ME.message);
        fclose('all');
        pause(0.5);
        if exist(h5_filename, 'file')
            delete(h5_filename);
        end
    end
end

fprintf('Creating new HDF5 file: %s ...\n', h5_filename);

% === Global data (real doubles) ===
rx_positions = [Xc, Yc, zf*ones(size(Xc))];
h5create(h5_filename, '/rx_positions', size(rx_positions));
h5write(h5_filename, '/rx_positions', rx_positions);

tx_pos = zeros(length(txs), 3);
tx_name_cell = cell(length(txs), 1);
for t = 1:length(txs)
    cTx = txs{t};
    tx_pos(t,:) = cTx.AntennaPosition(:)';
    tx_name_cell{t} = cTx.Name;
end

h5create(h5_filename, '/tx_positions', size(tx_pos));
h5write(h5_filename, '/tx_positions', tx_pos);

h5create(h5_filename, '/tx_names', size(tx_name_cell), 'Datatype', 'string');
h5write(h5_filename, '/tx_names', tx_name_cell);

% Steering vector
sv = phased.SteeringVector('SensorArray', antenna, 'PropagationSpeed', c);
num_ant = 16;

% === Process each transmitter ===
for tx_idx = 1:length(ray_results)
    data = ray_results{tx_idx};
    tx_group = sprintf('/tx_%d', tx_idx);
    
    fprintf('  Saving data for TX %d (%s) ...\n', tx_idx, tx_name_cell{tx_idx});
    
    % Real-valued scalar fields
    real_fields = {'path_loss', 'delay', 'gain_real', 'gain_imag', ...
                   'aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'doppler', 'rx_index'};

 % === 1. Save Real-Valued Fields (Path Loss, AoA, etc.) ===
    for f = 1:length(real_fields)
        fld = real_fields{f};
        if isfield(data, fld)
            % This trick extracts every field from every struct, 
            % turns them into columns (:), and stacks them vertically.
            temp_cell = {data.(fld)};
            val = cell2mat(cellfun(@(x) x(:), temp_cell, 'UniformOutput', false)');
            
            if ~isempty(val)
                ds_path = [tx_group '/' fld];
                % Standard HDF5 Create/Write
                try
                    h5create(h5_filename, ds_path, size(val));
                    h5write(h5_filename, ds_path, val);
                catch
                    h5write(h5_filename, ds_path, val);
                end
            end
        end
    end

% === 2. Save Complex Gains (Split into Real/Imag) ===
if isfield(data, 'gain_complex')
    g_cell = {data.gain_complex};
    % Force all to columns and stack
    all_gains = cell2mat(cellfun(@(x) x(:), g_cell, 'UniformOutput', false)');
    
    if ~isempty(all_gains)
        % Save Real Part
        ds_gr = [tx_group '/gain_complex_real'];
        try h5create(h5_filename, ds_gr, size(all_gains)); catch; end
        h5write(h5_filename, ds_gr, real(all_gains));
        
        % Save Imag Part
        ds_gi = [tx_group '/gain_complex_imag'];
        try h5create(h5_filename, ds_gi, size(all_gains)); catch; end
        h5write(h5_filename, ds_gi, imag(all_gains));
    end
end

% === Locations ===
% Wrap in [] to create a single matrix from the struct array
% Grab only the FIRST entry since TX location is constant for this group

one_tx_loc = data(1).tx_loc; 
if ~isempty(one_tx_loc)
    ds_tx = [tx_group '/tx_loc'];
    % Check if it exists (safety) before creating
    try
        h5create(h5_filename, ds_tx, size(one_tx_loc));
        h5write(h5_filename, ds_tx, one_tx_loc);
    catch
        % If it exists, just overwrite it
        h5write(h5_filename, ds_tx, one_tx_loc);
    end
end


    
% === RX Locations (Global Fingerprinting Version) ===
loc_cell = {data.rx_loc};
% Filter out the zeros from preallocation (rows that are [0 0 0])
valid_loc_mask = any(data.rx_loc ~= 0, 2);

if any(valid_loc_mask)
    % We want a 3 x TotalRays matrix for HDF5
    all_rx_loc = data.rx_loc(valid_loc_mask, :)'; 
    
    ds_rx = [tx_group '/rx_loc'];
    try
        h5create(h5_filename, ds_rx, size(all_rx_loc));
    catch
    end
    h5write(h5_filename, ds_rx, all_rx_loc);
    fprintf('  -> Successfully saved %d location points.\n', size(all_rx_loc, 2));
else
    fprintf('  ⚠️ Warning: No valid locations found in data.rx_loc for %s\n', tx_group);
end

% === Covariance Matrices & RSS ===

all_rx_indices = vertcat(data.rx_index); 
all_rx_indices = all_rx_indices(:);      
unique_rx_ids = unique(all_rx_indices);
num_rx_this_tx = length(unique_rx_ids);

% 1. Initialize Matrices AND RSS Array
cov_matrices_re = zeros(num_rx_this_tx, num_ant, num_ant, 'single');
cov_matrices_im = zeros(num_rx_this_tx, num_ant, num_ant, 'single');
rss_vals = zeros(num_rx_this_tx, 1, 'single'); % Initialize RSS array

% Safe flattening of rays
all_az_cell = {data.aoa_az};
all_el_cell = {data.aoa_el};
all_gains_cell = {data.gain_complex};

valid_r = ~cellfun(@isempty, all_az_cell);
all_az = cell2mat(cellfun(@(c) c(:), all_az_cell(valid_r), 'UniformOutput', false)');
all_el = cell2mat(cellfun(@(c) c(:), all_el_cell(valid_r), 'UniformOutput', false)');
all_gains = cell2mat(cellfun(@(c) c(:), all_gains_cell(valid_r), 'UniformOutput', false)');
    
for r = 1:num_rx_this_tx
    rx_id = unique_rx_ids(r);
    idx = all_rx_indices == rx_id;
    if sum(idx) == 0, continue; end
    
    az = all_az(idx);
    el = all_el(idx);
    g = all_gains(idx).'; 
    
    % --- RSS CALCULATION ---
    % Sum of power of all rays at this receiver (Linear scale)
    rss_lin = sum(abs(g).^2);
    % Convert to dBm (with a tiny floor to avoid log(0))
    rss_vals(r) = 10 * log10(rss_lin + 1e-20) + 30;
    
    % --- COVARIANCE CALCULATION ---
    A = sv(fq, [az(:)'; el(:)']);
        % This was  R = (A .* g) * A';
    B = A .* abs(g); %g is [1xnum_paths], A is [16xnum_paths] 
    R = B * B';          % Hermitian PSD
    
    if trace(R) > 0
        R = R / trace(R);
    end
    
    cov_matrices_re(r, :, :) = real(single(R));
    cov_matrices_im(r, :, :) = imag(single(R));
end

% --- SAVE EVERYTHING TO HDF5 ---
% Save Covariance Real
ds_cov_re = [tx_group '/covariance_real'];
h5create(h5_filename, ds_cov_re, size(cov_matrices_re), 'Datatype', 'single');
h5write(h5_filename, ds_cov_re, cov_matrices_re);

% Save Covariance Imaginary
ds_cov_im = [tx_group '/covariance_imag'];
h5create(h5_filename, ds_cov_im, size(cov_matrices_im), 'Datatype', 'single');
h5write(h5_filename, ds_cov_im, cov_matrices_im);

% Save RSS (New Dataset)
ds_rss = [tx_group '/rss'];
h5create(h5_filename, ds_rss, size(rss_vals), 'Datatype', 'single');
h5write(h5_filename, ds_rss, rss_vals);

fprintf('  TX %d: %d receivers -> Covariance and RSS saved.\n', tx_idx, num_rx_this_tx);    

% --- THE CLEANUP ---

ray_results{tx_idx} = [];   % Clear raw ray data from RAM
clear data rss_vals cov_matrices_re cov_matrices_im; % Clear temporary variables

end

fprintf('\n✅ All data saved successfully to: %s\n', h5_filename);
info = dir(h5_filename);
fprintf('File size: %.2f MB\n', info.bytes / 1e6);

toc;   % Show total time

% ====================== POST-RUN VERIFICATION ======================

fprintf('\n--- Running Data Integrity Check ---\n');
h5_info = h5info(h5_filename);

for t_idx = 1:length(h5_info.Groups)
    tx_grp = h5_info.Groups(t_idx).Name;
    
    % Read back a sample of the covariance
    cov_re = h5read(h5_filename, [tx_grp '/covariance_real']);
    
    % 1. Check for NaNs (indicates division by zero or empty rays)
    nan_count = sum(isnan(cov_re(:)));
    
    % 2. Check for All-Zeros (indicates receivers that received no signal)
    % We check if the sum of a matrix is 0 for any receiver
    zero_mats = 0;
    for r = 1:size(cov_re, 1)
        if sum(abs(squeeze(cov_re(r,:,:))), 'all') == 0
            zero_mats = zero_mats + 1;
        end
    end
    
    fprintf('Group %s:\n', tx_grp);
    fprintf('  -> Receivers: %d\n', size(cov_re, 1));
    if nan_count > 0
        fprintf('  ⚠️ WARNING: %d NaN values detected! Check trace(R) logic.\n', nan_count);
    end
    if zero_mats > 0
        fprintf('  ℹ️ NOTE: %d receivers have empty covariance (no rays reached them).\n', zero_mats);
    end
end

rx_positions = h5read('ray_data.h5', '/rx_positions');
fprintf('RX count: %d\n', size(rx_positions, 1));

% --- Paths per RX ---
fprintf('\n--- Paths per RX Statistics ---\n');
for t_idx = 1:length(h5_info.Groups)
    tx_grp = h5_info.Groups(t_idx).Name;
    try
        rx_idx_vec = h5read(h5_filename, [tx_grp '/rx_index']);
        rx_idx_vec = rx_idx_vec(:);
        
        unique_ids = unique(rx_idx_vec);
        counts = arrayfun(@(id) sum(rx_idx_vec == id), unique_ids);
        
        num_rx_reached = numel(unique_ids);
        num_rx_total   = size(rx_positions, 1);
        
        fprintf('  %s: %d/%d RX reached | paths/RX: mean=%.2f  max=%d  min=%d\n', ...
            tx_grp, num_rx_reached, num_rx_total, ...
            mean(counts), max(counts), min(counts));
        
        % Distribution
        for np = 1:min(max(counts), 10)
            pct = 100 * sum(counts == np) / num_rx_reached;
            if pct > 0.5
                fprintf('    %d path(s): %5.1f%% of RX\n', np, pct);
            end
        end
    catch ME
        fprintf('  %s: rx_index read failed — %s\n', tx_grp, ME.message);
    end
end


fprintf('--- Verification Complete ---\n');


function data = extract_ray_data(rays, current_tx, rxs, tx_idx, t_idx, velocity_vec, fc, offset, cart_pos)
    % offset: the starting index of this batch in the global all_rx array
    nRays = sum(cellfun(@length, rays));
    
    % --- Preallocate ---
    data.path_loss    = zeros(nRays,1);
    data.delay        = zeros(nRays,1);
    data.aoa_az       = zeros(nRays,1);
    data.aoa_el       = zeros(nRays,1);
    data.tx_loc       = zeros(nRays,3);
    data.rx_loc       = zeros(nRays,3);
    data.rx_index     = zeros(nRays,1);
    data.gain_complex = complex(zeros(nRays,1));

    tPos = current_tx.AntennaPosition(:)'; 
    idx = 1;

    for i = 1:length(rays)
        if ~isempty(rays{i})
            % Get RX Position from the site object for this batch element
            rPos = rxs(i).AntennaPosition(:)';
            
            for j = 1:length(rays{i})
                r = rays{i}(j);
                
                % --- Physics Data ---
                data.path_loss(idx) = r.PathLoss;
                data.delay(idx)     = r.PropagationDelay;
                data.aoa_az(idx)    = r.AngleOfArrival(1);
                data.aoa_el(idx)    = r.AngleOfArrival(2);
                
                % --- Global Indexing & Locations ---
                data.rx_index(idx)  = i + offset - 1;
                data.rx_loc(idx,:)  = rPos;
                data.tx_loc(idx,:)  = tPos;
                
                % --- Complex Gain (Phase: Deg to Rad) ---
                mag = 10^(-r.PathLoss/20);
                data.gain_complex(idx) = mag * exp(1j * deg2rad(r.PhaseShift));
                
                idx = idx + 1;
            end
        end
    end

    % --- Fill helper fields for the saving loop ---
    data.gain_real = real(data.gain_complex);
    data.gain_imag = imag(data.gain_complex);
    data.aod_az    = zeros(nRays,1);
    data.aod_el    = zeros(nRays,1);
    data.doppler   = zeros(nRays,1);
    data.time_idx  = ones(nRays,1) * t_idx;
end