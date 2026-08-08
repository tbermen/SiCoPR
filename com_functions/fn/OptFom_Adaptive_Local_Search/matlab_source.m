function skip_it = OptFom_Adaptive_Local_Search(LocalSearch_Value, BEST, THIS, FOM_history, iter_count, num_txffe_runs)

%%

%% --------------------------------------------------------
%  Logging config
%% --------------------------------------------------------
path_H= '';
log_csv_path = [path_H, 'ALS_log.csv'];  % <-- change this if you want a different file name
csv_header = {'iter','adaptive_radius','deterministic_radius','raw_L1_TX', ...
    'L1_w','L2_w','hard_cap', 'THIS.tx_index_vector', 'THIS.ctle_index','THIS.lp_idx',...
    'BEST.txffe_index', 'BEST.ctle','BEST.G_high_pass',...
	'THIS.vga_index', 'BEST.vga_index',...
    'THIS.FOM', 'BEST.FOM', 'FOM(end)','skip_it', 'skip_reason'};
log_yes_1_no_0= 0;

%% --------------------------------------------------------
%  Tuned knobs (PATCHED)
%% --------------------------------------------------------
min_improvement_threshold = 0.002;
adaptation_window         = 2;
radius_shrink_factor      = 0.60;   % *** PATCHED ***
deterministic_shrink_rate = 0.15;   % *** PATCHED ***
if num_txffe_runs== 1
    min_radius                = 1;
else
    min_radius                = 2;
end
min_radius= 1; % Forcing it to 1, if you set it to 2 then it tends to slow down
edge_weight = 1.0;
lp_weight   = 0.25;                 % *** PATCHED ***
vga_weight= 0.5;

l2_to_l1_ratio = 0.55;
use_hard_cap        = true;
hard_cap_multiplier = 1.2;



%% --------------------------------------------------------
%  Persistent memory
%% --------------------------------------------------------
persistent adaptive_radius no_improve_count initialized
skip_it = false;

if isempty(initialized) || iter_count == 1
    adaptive_radius   = max(min_radius, round(LocalSearch_Value));
    no_improve_count  = 0;
    initialized       = true;

    if log_yes_1_no_0== 1
        % Initialize CSV if missing
        if ~isfile(log_csv_path)
            append_csv_row(log_csv_path, csv_header, {});
        end
    end
end

%% --------------------------------------------------------
%  Read current FOM value
%% --------------------------------------------------------
if isempty(FOM_history)
    FOM = NaN;
else
    FOM = FOM_history(end);
end

%% --------------------------------------------------------
%  ADAPTIVE shrink (PATCHED)
%% --------------------------------------------------------
if numel(FOM_history) >= adaptation_window
    recent = FOM_history(end-adaptation_window+1:end);
    improvement = max(recent) - min(recent);

    if improvement < min_improvement_threshold
        no_improve_count = no_improve_count + 1;
    else
        no_improve_count = 0;
        adaptive_radius  = min(adaptive_radius + 1, LocalSearch_Value); % *** PATCHED ***
    end

    % shrink only when CTLE is "stable"
    if no_improve_count >= 1 && abs(THIS.ctle_index - BEST.ctle) <= 1
        adaptive_radius = max(min_radius, round(adaptive_radius * radius_shrink_factor));
        no_improve_count = 0;
    end
end

%% --------------------------------------------------------
%  Deterministic shrink (PATCHED)
%% --------------------------------------------------------
% deterministic_radius = round(LocalSearch_Value / (1 + deterministic_shrink_rate * iter_count));
% adaptive_radius = max(min_radius, min(adaptive_radius, deterministic_radius));

deterministic_radius = max(min_radius, round(LocalSearch_Value / (1 + deterministic_shrink_rate * iter_count)));
adaptive_radius      = max(min_radius, min(adaptive_radius, deterministic_radius));


%% --------------------------------------------------------
%  Extract tap vectors
%% --------------------------------------------------------
best_taps = BEST.txffe_index(:);
curr_taps = THIS.tx_index_vector(:);

ctle_index = THIS.ctle_index;
lp_curr    = THIS.g_LP_index;
lp_best    = BEST.G_high_pass;

if ~isfield(THIS, 'vga_index')
	vga_curr= 1;
else
	vga_curr = THIS.vga_index;
end

if ~isfield(BEST, 'vga_index')
	vga_best= 1;
else
	vga_best = BEST.vga_index;
end

if log_yes_1_no_0== 1
    if isempty(best_taps) || isempty(curr_taps)
        hard_cap = compute_hard_cap(use_hard_cap, hard_cap_multiplier, LocalSearch_Value, min_radius);
        append_csv_row(log_csv_path, csv_header, ...
            {iter_count, adaptive_radius, deterministic_radius, NaN, NaN, NaN, ...
            hard_cap, NaN, NaN, NaN,...
            NaN, NaN, NaN,...
			NaN, NaN,...
            THIS.FOM, BEST.FOM, FOM, double(skip_it), 'Skip: Empty BEST.txffe_index or THIS.tx_index_vector(:)!'});
        return;
    end
end

%% --------------------------------------------------------
%  Build weighted vectors
%% --------------------------------------------------------
best_vec = [best_taps; lp_best; vga_best];
this_vec = [curr_taps; lp_curr; vga_curr];

num_taps = numel(curr_taps);
w_taps = ones(num_taps,1) * edge_weight;

% LP weight (patched)
if THIS.ctle_index > 1
    w_lp = lp_weight * (1 + 0.5 * (THIS.ctle_index - 1));
else
    w_lp = lp_weight;
end

% VGA weight (patched)
% At higher CTLE: VGA impact increases.
if THIS.ctle_index > 1
    w_vga = vga_weight * (1 + 0.3*(THIS.ctle_index-1));
else
    w_vga = vga_weight;
end

weights = [w_taps; w_lp; w_vga];

%% --------------------------------------------------------
%  Compute distances
%% --------------------------------------------------------
diff_vec      = this_vec - best_vec;
weighted_diff = weights .* diff_vec;

L1_w = sum(abs(weighted_diff));
L2_w = sqrt(sum(weighted_diff.^2));

raw_L1_TX = sum(abs(diff_vec(1:num_taps)));

%% --------------------------------------------------------
%  CTLE constraint (PATCHED to Â±2)
%% --------------------------------------------------------
if abs(THIS.ctle_index - BEST.ctle) > 2
    skip_it = true;

    if log_yes_1_no_0== 1
        hard_cap = compute_hard_cap(use_hard_cap, hard_cap_multiplier, LocalSearch_Value, min_radius);
        append_csv_row(log_csv_path, csv_header, ...
            {iter_count, adaptive_radius, deterministic_radius, ...
            raw_L1_TX, L1_w, L2_w, hard_cap, mat2str(curr_taps), ctle_index, lp_curr,...
            mat2str(best_taps), BEST.ctle, lp_best,...
			THIS.vga_index, BEST.vga_index,...
            THIS.FOM, BEST.FOM, FOM, double(skip_it), 'Skip: CTLE_idx too far!'});
    end
    return;
end

%% --------------------------------------------------------
%  Early exact match
%% --------------------------------------------------------
if L1_w == 0
    skip_it = false;
    if log_yes_1_no_0== 1
        hard_cap = compute_hard_cap(use_hard_cap, hard_cap_multiplier, LocalSearch_Value, min_radius);
        append_csv_row(log_csv_path, csv_header, ...
            {iter_count, adaptive_radius, deterministic_radius, ...
            raw_L1_TX, L1_w, L2_w, hard_cap, mat2str(curr_taps), ctle_index, lp_curr,...
            mat2str(best_taps), BEST.ctle, lp_best,...
			THIS.vga_index, BEST.vga_index,...
            THIS.FOM, BEST.FOM, FOM, double(skip_it), 'Evaluate Candidate: Exact Match!'});
    end
    return;
end

%% --------------------------------------------------------
%  Hard cap
%% --------------------------------------------------------
hard_cap = compute_hard_cap(use_hard_cap, hard_cap_multiplier, LocalSearch_Value, min_radius);

if use_hard_cap && raw_L1_TX > hard_cap
    skip_it = true;
    if log_yes_1_no_0== 1
        append_csv_row(log_csv_path, csv_header, ...
            {iter_count, adaptive_radius, deterministic_radius, ...
            raw_L1_TX, L1_w, L2_w, hard_cap, mat2str(curr_taps), ctle_index, lp_curr,...
            mat2str(best_taps), BEST.ctle, lp_best,...
			THIS.vga_index, BEST.vga_index,...
            THIS.FOM, BEST.FOM, FOM, double(skip_it), 'Skip: TX Exceeds Cap!'});
    end
    return;
end

%% --------------------------------------------------------
%  L1/L2 skip rule
%% --------------------------------------------------------
L2_threshold = max(min_radius, ceil(l2_to_l1_ratio * adaptive_radius));

if (L1_w > adaptive_radius) && (L2_w > L2_threshold)
    skip_it = true;
else
    skip_it = false;
end

%% --------------------------------------------------------
%  Final log write
%% --------------------------------------------------------
if log_yes_1_no_0
    if skip_it== 0
        append_csv_row(log_csv_path, csv_header, ...
            {iter_count, adaptive_radius, deterministic_radius, ...
            raw_L1_TX, L1_w, L2_w, hard_cap, mat2str(curr_taps), ctle_index, lp_curr,...
            mat2str(best_taps), BEST.ctle, lp_best,...
			THIS.vga_index, BEST.vga_index,...
            THIS.FOM, BEST.FOM, FOM, double(skip_it), 'Evaluate Candidate'});
    else
        append_csv_row(log_csv_path, csv_header, ...
            {iter_count, adaptive_radius, deterministic_radius, ...
            raw_L1_TX, L1_w, L2_w, hard_cap, mat2str(curr_taps), ctle_index, lp_curr,...
            mat2str(best_taps), BEST.ctle, lp_best,...
			THIS.vga_index, BEST.vga_index,...
            THIS.FOM, BEST.FOM, FOM, double(skip_it), 'Skip: Outside L1/ L2 Limits!'});
    end
end
