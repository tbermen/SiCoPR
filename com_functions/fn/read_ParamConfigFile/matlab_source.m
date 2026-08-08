function [param,OP]= read_ParamConfigFile(paramFile,OP)
%warning('off','MATLAB:xlsread:Mode'); % suppress warning messages for reading the settings file from XLS
[filepath,name,ext] = fileparts(paramFile);
if ~isempty(filepath)
    filepath=[filepath '\'];
end
matcongfile=[filepath name '.mat'];
try
    switch upper(ext)
        case upper('.mat')
            load(matcongfile)
        case upper('.csv')
            [na1, na2, parameter] = xlsread(paramFile);
        otherwise
            [na1, na2, parameter] = xlsread(paramFile,'COM_Settings','',''); %#ok<ASGLU> % Import data from the settings file (imports the entire sheet)
    end
    
catch ME %#ok<NASGU>
    warning('off','MATLAB:xlsread:Mode'); % suppress warning messages for reading the settings file from XLS
    switch upper(ext)
        case upper('.mat')
            load(matcongfile)
        case upper('.csv')
            [na1, na2, parameter] = xlsread(paramFile);
        otherwise
            [na1, na2, parameter] = xlsread(paramFile,'COM_Settings','','basic'); %#ok<ASGLU> % Import data from the settings file (imports the entire sheet)
    end
end
OP.SAVE_KEYWORD_FILE = xls_parameter(parameter, 'SAVE_KEYWORD_FILE', false, 0); % Save csv file of COM parameter (OP) and keywords. Not implemented.


%% New section to parse .START package data
first_column_data = parameter(:,1);
start_data_rows = find(strcmp(first_column_data,'.START'));
if ~isempty(start_data_rows)
    end_data_rows = find(strcmp(first_column_data,'.END'));
    if length(start_data_rows) ~= length(end_data_rows)
        error('Number of .START and .END must be the same');
    end
    first_start_row = start_data_rows(1);
    special_parameter = parameter;
    parameter = parameter(1:first_start_row-1,:);
    for j=1:length(start_data_rows)
        this_block = special_parameter(start_data_rows(j)+1:end_data_rows(j)-1,:);
        pkg_name = special_parameter{start_data_rows(j),2};
        
        %Read all the parameters that make up a package
        PKG_param = read_package_parameters(this_block);
        
        %save the data in a field revealed by pkg_name
        param.PKG.(pkg_name) = PKG_param;
        
        
    end
end
%Allow specification of TX and RX package section through PKG_NAME keyword
%the values must match package blocks specified in .START sections
param.PKG_NAME= xls_parameter(parameter, 'PKG_NAME', false,'');
if isnan(param.PKG_NAME)
    param.PKG_NAME = '';
end
if isempty(param.PKG_NAME)
    param.PKG_NAME = {};
else
    param.PKG_NAME = strsplit(param.PKG_NAME);
end
if ~isempty(param.PKG_NAME) && ~isfield(param,'PKG')
    error('PKG_NAME can only be used if .START blocks for package parameters are used');
end
for j=1:length(param.PKG_NAME)
    if ~isfield(param.PKG,param.PKG_NAME{j})
        error('Package Block "%s" not found',param.PKG_NAME{j});
    end
end

%%
% just need to define so we can pass
param.c=[.4e-12 .4e-12]; 
param.alen=[ 20 30 550  ];
param.az=[100 120 100];

% make control for package/channel reflection control
param.kappa1= xls_parameter(parameter, 'kappa1', true,1); % if set 0 reflection at tp0 are omitted from COM
param.kappa2= xls_parameter(parameter, 'kappa2', true,1); % if set 0 reflection at tp5 are omitted from COM

% make compatible with presentation of kappa
% Default values are given for parameters when they are common to all clauses in 802.3bj and 803.2bm.

OP.dynamic_txffe = xls_parameter(parameter, 'Dynamic TXFFE', false,1); % Temporary switch for testing new optimize_fom with dynamic txffe
OP.FloatingDFE_Development = xls_parameter(parameter, 'FloatingDFE_Development', false,1); % Temporary switch for testing new floating dfe routine

param.fb = xls_parameter(parameter, 'f_b')*1e9; % Baud (Signaling) rate in Gbaud
param.f2 = xls_parameter(parameter, 'f_2', true, param.fb/1e9 )*1e9; % frequency in GHz for intergration compuation of ICN GHz
param.f2_ild = xls_parameter(parameter, 'f_2_ILD', true, param.f2/1e9 )*1e9; % frequency in GHz for intergration compuation FOM_Ild in GHz. f2 is used if not specified
param.max_start_freq = xls_parameter(parameter, 'f_min')*1e9; % minimum required frequency start for s parameters
param.f1 = xls_parameter(parameter, 'f_1', true, param.max_start_freq/1e9)*1e9; % start frequency for ICN and ILD calculations in GHz
param.max_freq_step = xls_parameter(parameter, 'Delta_f')*1e9; % freqency step
param.tx_ffe_c0_min = xls_parameter(parameter, 'c(0)', false); % TX equalizer cursor minimum value (actual value is calculated as 1-sum(abs(tap)), Grid seat ingored for when C(0) is below this value

if OP.dynamic_txffe
    found_pre=1;
    pre_count=1;
    while found_pre
        [p,found_pre]=xls_parameter_txffe(parameter,sprintf('c(-%d)',pre_count));
        if found_pre
            field_name=sprintf('tx_ffe_cm%d_values',pre_count);
            param.(field_name)=p;
            pre_count=pre_count+1;
        end
    end
    found_post=1;
    post_count=1;
    while found_post
        [p,found_post]=xls_parameter_txffe(parameter,sprintf('c(%d)',post_count));
        if found_post
            field_name=sprintf('tx_ffe_cp%d_values',post_count);
            param.(field_name)=p;
            post_count=post_count+1;
        end
    end
else
param.tx_ffe_cm4_values = xls_parameter(parameter, 'c(-4)', true,0); % TX equalizer pre cursor tap -4 individual settings or range. If not present ignored
param.tx_ffe_cm3_values = xls_parameter(parameter, 'c(-3)', true,0); % TX equalizer pre cursor tap -3 individual settings or range. If not present ignored
param.tx_ffe_cm2_values = xls_parameter(parameter, 'c(-2)', true,0); % TX equalizer pre cursor tap -2 individual settings or range. If not present ignored
param.tx_ffe_cm1_values = xls_parameter(parameter, 'c(-1)', true,0); % TX equalizer pre cursor tap -1 individual settings or range. If not present ignored
param.tx_ffe_cp1_values = xls_parameter(parameter, 'c(1)', true,0); % TX equalizer post cursor tap 1  individual settings or range. If not present ignored
param.tx_ffe_cp2_values = xls_parameter(parameter, 'c(2)', true,0); % TX equalizer post cursor tap 2  individual settings or range. If not present ignored
param.tx_ffe_cp3_values = xls_parameter(parameter, 'c(3)', true,0); % TX equalizer post cursor tap 3  individual settings or range. If not present ignored
end
param.ndfe = xls_parameter(parameter, 'N_b'); % Decision feedback fixed equalizer (DFE) length
param.N_v = xls_parameter(parameter, 'N_v',true,param.ndfe); % number of UI used to compute Vf
param.D_p = xls_parameter(parameter, 'D_p',true, 4 ); % number of precursor UI's used to compute Vf Default to 4
param.N_p = xls_parameter(parameter, 'N_p',true, 400 ); % number of precursor UI's used to compute Vf Default to  400
param.N_bx = xls_parameter(parameter, 'N_bx', true, param.ndfe ); % Used for ERL to Compensate for a number of Ui assoicated with the DFE
% support for floating taps
param.N_bg=xls_parameter(parameter, 'N_bg', true,0); % number of group of floating tap. Used as a switch, 0 means no float
param.N_bf=xls_parameter(parameter, 'N_bf', true,6); % number of taps in group
param.N_bmax=xls_parameter(parameter, 'N_bmax', true, param.ndfe); % UI span for floating taps
param.N_bmax=xls_parameter(parameter, 'N_f', true, param.ndfe); % UI span for floating taps. replaced by N_bmax
param.N_f=xls_parameter(parameter, 'N_f', true, param.ndfe); % UI span for floating taps for rX FFE
if param.N_bg == 0, param.N_bmax=param.ndfe; end
param.bmaxg=xls_parameter(parameter, 'bmaxg', true,0.2); % max DFE value for floating taps

% support for tail tap power limitations
param.B_float_RSS_MAX=xls_parameter(parameter, 'B_float_RSS_MAX', true,0); % floating DFE tap start for RSS floating tap limit
param.N_tail_start=xls_parameter(parameter, 'N_tail_start', true,0); % start range for max RSS limit for DFE taps
%
param.dfe_delta = xls_parameter(parameter, 'N_b_step', true,0); % discreatiztion of DFE, 0 disables and is not normally used
param.ffe_pre_tap_len=xls_parameter(parameter, 'ffe_pre_tap_len', true,0); % RX ffe pre cursor tap length
param.RxFFE_cmx=param.ffe_pre_tap_len;
param.ffe_post_tap_len=xls_parameter(parameter, 'ffe_post_tap_len', true,0); % Rx FFE post cursor tap length
param.RxFFE_cpx=param.ffe_post_tap_len;
param.ffe_tap_step_size=xls_parameter(parameter, 'ffe_tap_step_size', true,0); % Rx FFE tap step size
param.RxFFE_stepz=param.ffe_tap_step_size;
param.ffe_main_cursor_min=xls_parameter(parameter, 'ffe_main_cursor_min', true,1); % Rx FFE main cursor miminum 
param.ffe_pre_tap1_max=xls_parameter(parameter, 'ffe_pre_tap1_max', true,.7); % Rx FFE precursor tap1 limit
param.ffe_post_tap1_max=xls_parameter(parameter, 'ffe_post_tap1_max', true,.7); % Rx FFE post cursor tap1 limit
param.ffe_tapn_max=xls_parameter(parameter, 'ffe_tapn_max', true,.7); % Rx FFE precursor tapn limit
param.ffe_backoff=xls_parameter(parameter, 'ffe_backoff', true,0); % see if better zero foreced solution is better by backing off the number specified FFE taps one at at time
if param.RxFFE_cmx ~= 0 || param.RxFFE_cpx ~=0
    OP.RxFFE= true;
else
    OP.RxFFE=false;
end
param.num_ui_RXFF_noise=xls_parameter(parameter, 'num_ui_RXFF_noise', true,2048); % Rx FFE precursor tapn limit
param.flim=xls_parameter(parameter, 'flim', true,1000e9); % limit sparameter freq range RIM 05/30/2025
if param.flim <1000 param.flim=param.flim*1e9;end % allow specification in GHz 
param.g_DC_HP_values = xls_parameter(parameter, 'g_DC_HP', true,[]); % CTF AC-DC gain list (GDC2)
param.f_HP = 1e9*xls_parameter(parameter, 'f_HP_PZ', true, []); % CFT pole pole zero pair in GHz for low frequency CTF
param.f_HP_Z = 1e9*xls_parameter(parameter, 'f_HP_Z', true, []); % CFT zero fz1 is in GHz. Normally a list for 120e. Not normally use elsewise
param.f_HP_P = 1e9*xls_parameter(parameter, 'f_HP_P', true, []); % CFT pole fp2 is in GHz. Normally a list for 120e. Not normally use elsewise


param.Min_VEO= xls_parameter(parameter, 'EH_min', true,0); % used when PMD_type is C2M
param.Max_VEO= xls_parameter(parameter, 'EH_max', true,inf); % used when PMD_type is C2M and is not really computed per spec
param.Min_VEO_Test= xls_parameter(parameter, 'EH_min_test', true,0); % Older syntax. Used when PMD_type is C2M. This allow EH to go below EH_min. If set to zero it is ignored (same as Min_VEO_test)
param.Min_VEO_Test= xls_parameter(parameter, 'Min_VEO_Test', true,param.Min_VEO_Test); % used when PMD_type is C2M. This allow EH to go blow EH_min. If set to Zero it is ignored

param.CTLE_type= xls_parameter(parameter, 'CTLE_type', false,'CL93'); % Sets the CTLE type default is poles and zeros (i.e. not a list of poles as in 120e) 
if ~isempty(param.g_DC_HP_values) ; param.CTLE_type='CL120d';end % overrides CL93 if g_DC_HP_values are a spreadsheet entry. Mostly used when baud rare is >= 50Gbps
if ~isempty(param.f_HP_Z) ; param.CTLE_type='CL120e';end % overrides CL93 and CL120d if f_HP_Z is a spreadsheet entry
% always read in main ctle values. They would be interpreted different baseed
% on the clause they apply because of different CTF equations
param.ctle_gdc_values = xls_parameter(parameter, 'g_DC', true); % AC-DC gain list
param.CTLE_fp1 = 1e9*xls_parameter(parameter, 'f_p1', true, param.fb/4); % fp1 is in GHz
param.CTLE_fp2 = 1e9*xls_parameter(parameter, 'f_p2', true, param.fb); % fp2 is in GHz
param.CTLE_fz = 1e9*xls_parameter(parameter, 'f_z', true, param.fb/4); % fz is in GHz
% the contex of the poles an zeros are determined by the clause
switch param.CTLE_type
    case 'CL93'
        param.ctle_gdc_values = xls_parameter(parameter, 'g_DC', true); % Continuous time filter DC gain settings (G_DC) or range as specified in Annex 93A
        param.CTLE_fp1 = 1e9*xls_parameter(parameter, 'f_p1', true, param.fb/4); % fp1 is in GHz
        param.CTLE_fp2 = 1e9*xls_parameter(parameter, 'f_p2', true, param.fb); % fp2 is in GHz
        param.CTLE_fz = 1e9*xls_parameter(parameter, 'f_z', true, param.fb/4); % fz is in GHz
    case 'CL120d'
        param.g_DC_HP_values = xls_parameter(parameter, 'g_DC_HP', true,[]); % Continuous time filter DC gain settings (G_DC2)
        param.f_HP = 1e9*xls_parameter(parameter, 'f_HP_PZ', true, []); % fLF is in GHz
    case 'CL120e'
        % re adjust to get TD_CTLE to work with C:120e equation without
        % changing TD_CTLE code
        param.CTLE_fz =param.CTLE_fz ./ 10.^(param.ctle_gdc_values/20);
end
param.GDC_MIN = xls_parameter(parameter, 'GDC_MIN',true, 0); % max ACDC gain, if 0 ignore
param.cursor_gain=xls_parameter(parameter, 'crusor_gain', true,0); % only FFE and not supported
%% addd default to support multiple packages
param.a_thru = xls_parameter(parameter, 'A_v', true, 0.5); % Victim differential peak source output voltage (half of peak to peak)
param.a_fext = xls_parameter(parameter, 'A_fe', true,0.5); % FEXT aggressor differential peak source output voltage (half of peak to peak)
param.a_next = xls_parameter(parameter, 'A_ne', true,0.5); % NEXT aggressor differential peak source output voltage (half of peak to peak)
param.a_icn_fext = xls_parameter(parameter, 'A_ft', true, param.a_fext); % FEXT aggressor amplitude for ICN. Defaults to A_fe if not specified
param.a_icn_next = xls_parameter(parameter, 'A_nt', true, param.a_next );% NEXT aggressor amplitude for ICN. Defaults to A_ne if not specified
param.levels = xls_parameter(parameter, 'L'); % number of symbols levels (PAM-4 is 4, NRZ is 2)
param.specBER = xls_parameter(parameter, 'DER_0'); % Target detector error ratio
param.DER_CDR = xls_parameter(parameter, 'DER_CDR',true,1e-2); % min DER required for a CDR 
param.N_qb = xls_parameter(parameter, 'N_qb',true,0); % adc number of bits if 0 do not apply quantization
param.P_qc= xls_parameter(parameter, 'P_qc',true,2*param.specBER); % adc clipping probability
param.clip_method = xls_parameter(parameter, 'Clip Method', false, 'Fast'); % "Clip Method" parameter for Quantization Noise ('Fast' or 'Slow')
param.P_peak= xls_parameter(parameter, 'P_peak',true,param.specBER); % SCMR CDFprobability. Default DER_0 
param.pass_threshold = xls_parameter(parameter, 'COM Pass threshold',false,0); % the pass fail threshold for COM in dB
param.add_rx_noise = xls_parameter(parameter, 'add_rx_noise', true, param.pass_threshold); % additional receiver noise target in dB
param.ERL_pass_threshold = xls_parameter(parameter, 'ERL Pass threshold',false,0); % the pass fail threshold for ERL in dB
param.VEC_pass_threshold = xls_parameter(parameter, 'VEC Pass threshold',false,0);% the pass fail threshold for VEC in dB only used when PMD_type is C2M

param.sigma_RJ = xls_parameter(parameter, 'sigma_RJ'); % rms of of random jitter
param.A_DD = xls_parameter(parameter, 'A_DD'); % Normalized peak dual-Dirac noise, this is half of the total bound uncorrelated jitter (BUJ) in UI
param.eta_0 = xls_parameter(parameter, 'eta_0'); % One-sided noise spectral density (V^2/GHz).Input refered noise at TP5. Input referred noise at TP5
% shakiba_3dj_01_2407 (to add MLSE sequence truncation penalty)
param.trunc = xls_parameter(parameter, 'trunc', true, 128); % MLSE sequence truncation length
param.trunc = xls_parameter(parameter, 'N_tc', true, param.trunc); % MLSE sequence truncation length

% healey_3dj_01a_2407 adjust MLSE for COM_DFE
param.Q_budget_adj = xls_parameter(parameter, 'Q_budget_adj', true, 0); % MLSE sequence truncation length
param.SNDR = xls_parameter(parameter, 'SNR_TX', true); % Transmitter SNDR noise in dB
param.S_tn_w_AM = xls_parameter(parameter, 'S_tn_w_AM', false, 0); % 0/1, remove sigma_X2 if param.S_tn_w_AM = 0 else it's 1 and will use sigma_X2
param.R_LM = xls_parameter(parameter, 'R_LM'); % Ratio of level separation mismatch. Relevant when not PAM-2 (NRZ).
param.samples_per_ui = xls_parameter(parameter, 'M', 32); % Samples per UI
param.ts_sample_adj_range = xls_parameter(parameter, 'sample_adjustment', true, [0 0]); %sample point adjustment range
param.ts_anchor = xls_parameter(parameter, 'ts_anchor', true, 0); %choice of sampling routine.  0=MM, 1=Peak, 2=max DV (max cursor minus precursor)
% This will keep bmax length 0 if Nb=0

%AJG021820
param.bmax(1:param.ndfe)     = xls_parameter(parameter, 'b_max(1)'); % DFE magnitude limit, first coefficient(ignored if Nb=0)
if isempty(param.bmax)
    param.bmin=param.bmax;
else
    param.bmin(1:param.ndfe)     =xls_parameter(parameter, 'b_min(1)', true,-param.bmax(1) ); % DFE negative magnitude limit. If not specified it defaults to -bmax.

end
if param.ndfe >= 2
    param.bmax(2:param.ndfe) = xls_parameter(parameter, 'b_max(2..N_b)', true, .2); % DFE magnitude limit, second coefficient and on (ignored if Nb<2). Can be a regualar expression
    param.bmin(2:param.ndfe) = xls_parameter(parameter, 'b_min(2..N_b)', true, -1*param.bmax(2:param.ndfe) ); % DFE negative magnitude limit, if not specified it defaults to -b_max(2..N_b)
end

param.gqual=xls_parameter(parameter, 'G_Qual', true,[]);% G_Qual are the dB ranges of g_DC g DC )which correspond tog_DC_HP (g DC2)
param.g2qual=xls_parameter(parameter, 'G2_Qual', true,[]); % G2_Qual limit values of g_DC_HP (g DC2 ) which corresponds to ranges of g_DC g DC specified with G_QUAL
%verify gqual and gqual2 input
if ~isempty(param.gqual) || ~isempty(param.g2qual)
    if size(param.gqual,1)~=length(param.g2qual)
        error('gqual and g2qual size mismatch');
    end
    if size(param.gqual,2)~=2
        error('gqual must be Nx2 matrix');
    end
end


% eval if string for all three - can use different for TX and RX
param.C_pkg_board = xls_parameter(parameter, 'C_p', true,0)*1e-9; % C_p in nF (single sided)
param.C_diepad = xls_parameter(parameter, 'C_d', true,0)*1e-9; % C_d in nF (single sided)
% [ahealey] Read values for optional compensating L and "bump" C
param.L_comp = xls_parameter(parameter, 'L_s', true, 0)*1e-9; % L_s in nH (single sided)
param.C_bump = xls_parameter(parameter, 'C_b', true, 0)*1e-9; % C_b in nF (single sided)
% [ahealey] End of modification
param.Z0 = xls_parameter(parameter, 'R_0', 50); % 
% added default to support multiple packages
param.C_v = xls_parameter(parameter, 'C_v', true,0)*1e-9; % C_v in nF (via cap)  (single sided)
param.R_diepad = xls_parameter(parameter, 'R_d', true, [50,50]); % Die source termination resistance  (single sided)
param.Z_t = xls_parameter(parameter, 'Z_t', true,param.Z0); %  single sided source termination reference resistance for TDR and ERL 
param.TR_TDR = xls_parameter(parameter, 'TR_TDR', true , 8e-3); %  Gaussian shaped transition time for TDR source in ns 



% added default to support multiple packages
param.z_p_tx_cases = xls_parameter(parameter, 'z_p (TX)', true,'[ 8  24 30 45 ; 1  1   1 1;  1 1  1  1 ; 0.5 0.5  0.5  0.5   ]').'; % List of victim transmitter package trace lengths in mm, one per case
[ncases, mele]=size(param.z_p_tx_cases);
if mele ==2
    param.flex=2;
elseif mele==4
    param.flex=4;
elseif mele==1
    param.flex=1;
else
    error(sprintf('config file syntax error'))
end

% board parameters
param.C_0 = xls_parameter(parameter, 'C_0', true,0)*1e-9; % If Include PCB is set to 1, near device single ended capacitance C0  in nF is added  
param.C_1 = xls_parameter(parameter, 'C_1', true,0)*1e-9; % if Include PCB is set to 1, connector side single ended capacitance C1 in nF is added 
% added default to support multiple packages
param.z_p_next_cases = xls_parameter(parameter, 'z_p (NEXT)', true,'[ 8  24 30 45 ; 1  1   1 1;  1 1  1  1 ; 0.5 0.5  0.5  0.5   ]').'; % List of NEXT transmitter package trace lengths in mm, one per case
[ncases1, mele1]=size(param.z_p_next_cases);
if ncases ~= ncases1 || mele ~= mele1
    error('All TX, NEXT, FEXT, Rx cases must agree');
else
end
param.z_p_fext_cases = xls_parameter(parameter, 'z_p (FEXT)', true,'[ 8  24 30 45 ; 1  1   1 1;  1 1  1  1 ; 0.5 0.5  0.5  0.5   ]').'; % List of FEXT transmitter package trace lengths in mm, one per case
[ncases1, mele1]=size(param.z_p_fext_cases);
if ncases ~= ncases1 ||  mele ~= mele1
    error('All TX, NEXT, FEXT, Rx cases must agree');
else
end
param.z_p_rx_cases = xls_parameter(parameter, 'z_p (RX)', true,'[ 8  24 30 45 ; 1  1   1 1;  1 1  1  1 ; 0.5 0.5  0.5  0.5   ]').'; % List of FEXT receiver package trace lengths in mm, one per case
[ncases1, mele1]=size(param.z_p_rx_cases);
if ncases ~= ncases1 ||  mele ~= mele1
    error('All TX, NEXT, FEXT, Rx cases must agree');
else
end
% Table 93A-3 parameters
param.pkg_gamma0_a1_a2 = xls_parameter(parameter, 'package_tl_gamma0_a1_a2', true, [0 1.734e-3 1.455e-4]); %Fitting parameters for package model per unit length. First element is in 1/mm and affects DC loss of package model . Second element is in ns1/2/mm and affects loss proportional to sqrt(f). Third element is in ns/mm and affects loss proportional to f.
param.pkg_tau = xls_parameter(parameter, 'package_tl_tau', true, 6.141e-3); % Package model transmission line delay ns/mm
param.pkg_Z_c = xls_parameter(parameter, 'package_Z_c', true, '[92 92 ; 70 70; 80 80; 100 100]').';% Package model transmission line characteristic impedance [ Tx , Rx ]
[ ncases1, mele1]=size(param.pkg_Z_c);% 
if   mele ~= mele1
    error('tx rx pairs must have thesame number element entries as TX, NEXT, FEXT, Rx');
else
end
if mele1==2 % fuill in a array if only a 2 element flex package is specified
    for ii=1:ncases
        param.z_p_fext_casesx(ii,:)=  [param.z_p_fext_cases(ii,:)' ;[ 0 ; 0 ]]';
        param.z_p_next_casesx(ii,:)=  [param.z_p_next_cases(ii,:)' ;[ 0 ; 0 ]]';
        param.z_p_tx_casesx(ii,:)=  [param.z_p_tx_cases(ii,:)' ;[ 0 ; 0 ]]';
        param.z_p_rx_casesx(ii,:)=  [param.z_p_rx_cases(ii,:)' ;[ 0 ; 0 ]]';
    end
    param.z_p_fext_cases =  param.z_p_fext_casesx;
    param.z_p_next_cases=  param.z_p_next_casesx;
    param.z_p_tx_cases=  param.z_p_tx_casesx;
    param.z_p_rx_cases=  param.z_p_rx_casesx;
    param.pkg_Z_c=[param.pkg_Z_c' ;[ 100 100 ; 100 100 ]]';
end
param.PKG_Tx_FFE_preset =xls_parameter(parameter, 'PKG_Tx_FFE_preset', true, 0); % RIM 08-18-2022 for Tx preset capability

% Table 92-12 parameters
param.brd_gamma0_a1_a2 = xls_parameter(parameter, 'board_tl_gamma0_a1_a2', true, [0 4.114e-4 2.547e-4]); % Fitting parameters for package model per unit length. First element is in 1/mm and affects DC loss of package model . Second element is in ns1/2/mm and affects loss proportional to sqrt(f). Third element is in ns/mm and affects loss proportional to f.
param.brd_tau = xls_parameter(parameter, 'board_tl_tau', true, 6.191e-3);% Board model transmission line delay ns/mm
param.brd_Z_c = xls_parameter(parameter, 'board_Z_c', true, 109.8); % Board model transmission line characteristic impedance [ Tx , Rx ]
param.z_bp_tx = xls_parameter(parameter, 'z_bp (TX)', true, 151); %  Victim transmitter board trace lengths in mm
param.z_bp_next = xls_parameter(parameter, 'z_bp (NEXT)', true, 72);% Next Assessor transmitter board trace lengths in mm
param.z_bp_fext = xls_parameter(parameter, 'z_bp (FEXT)', true, 72);% Rext Assessor transmitter board trace lengths in mm
param.z_bp_rx = xls_parameter(parameter, 'z_bp (RX)', true, 151);% Victim receiver board trace lengths in mm

% Unofficial parameters
param.snpPortsOrder = xls_parameter(parameter, 'Port Order', true, [1 3 2 4]); % s parameter port order [ tx+ tx- rx+ rx-]
param.delta_IL=xls_parameter(parameter, 'delta_IL', false, 1); % experiemnal
% Deprecated parameters - affect only frequency domain analysis.
param.f_v = xls_parameter(parameter, 'f_v', true, 4); % For FOM_ILD: Transiton rate cut off frequency for ICN/ILD calc in terms of fb
param.f_f = xls_parameter(parameter, 'f_f', true, 4); % For ICN: Fext transiton rate cut off frequency for ICN calc in terms of fb
param.f_n = xls_parameter(parameter, 'f_n', true, 4); % For ICN: Next transiton rate cut off frequency for ICN calc in terms of fb
param.f_r = xls_parameter(parameter, 'f_r', true, 4); % reference receive filter in COM and in ICN/FOM_ILD calcs in terms of fb
param.fb_BT_cutoff= xls_parameter(parameter, 'TDR_f_BT_3db', true, 0.4730); % Bessel-Thomson 3 dB cut off freqeuncy in terms of fb
param.BTorder =  xls_parameter(parameter, 'BTorder', false, 4); % Bessel function order
param.RC_Start =  xls_parameter(parameter, 'RC_Start', false, param.fb/2); % start frequency for raised cosine filter
param.RC_end =  xls_parameter(parameter, 'RC_end', false, param.fb*param.f_r ); % end frequency for raised cosine filter
param.beta_x= xls_parameter(parameter, 'beta_x', false, 0);%  (for ERL) use default
param.rho_x= xls_parameter(parameter, 'rho_x', false, .618); % (for ERL) use default
param.tfx= xls_parameter(parameter, 'fixture delay time', true, -1);% fixture delay time (for ERL)
param.Grr_limit=xls_parameter(parameter, 'Grr_limit', false, 1); % either do no use or set to 1 (for ERL)
param.Grr=xls_parameter(parameter, 'Grr', false, param.Grr_limit);% either do no use or set to 1 (for ERL)
param.Gx=xls_parameter(parameter, 'Gx', false, 0); % ERL parameter param.Grr, This is used is the COM code
switch param.Gx
    case 0
        param.Grr=param.Grr; % just use older Grr ir gx not specified
    case 1
        param.Grr=2; % use newer Grr
end

param.LOCAL_SEARCH=xls_parameter(parameter,'Local Search',true,0); % Decreases COM compute time. Aetting to 2 seems ok ,if 0 search is full grid
% Operational control variables
%OP.include_pcb = xls_parameter(parameter, 'Include PCB (table 92-13)', false, 0);
param.Tukey_Window=xls_parameter(parameter,'Tukey_Window',true,0); % required for ERL. Set to 1. Default is 0.
param.zero_pad_tukey_window_in_fb=xls_parameter(parameter,'zero_pad_tukey_window_in_fb',true,0); % window relative between fstop and fb as a percentage of fb
param.Noise_Crest_Factor= xls_parameter(parameter, 'Noise_Crest_Factor', true, 0); % Normally not used. If set this is q factor used for quantized Gaussian PDFs
param.AC_CM_RMS = xls_parameter(parameter, 'AC_CM_RMS', true, 0); % AC_CM_RMS is the CM BBN AWGN RMS at COM source point. Default is 0. Adds common mode noise source to the COM signal path for the through channel
param.ACCM_MAX_Freq=xls_parameter(parameter, 'ACCM_MAX_Freq', true, param.fb); % F max for integrating ACCM voltage in Hz. Default is fb
param.T_O = xls_parameter(parameter, 'T_O', true, 0 ); % Units are mUI. Histogram for VEC and VEO are computed over T_s +/- T_O.  
param.T_O = xls_parameter(parameter, 'T_h', true, param.T_O  ); % superceded with T_O but is the internal values that is used. Do not use.

param.samples_for_C2M =xls_parameter(parameter, 'samples_for_C2M', true, 100 ); % Finer sampling in terms of samples per UI for c2m histgram analysis.

OP.Histogram_Window_Weight=xls_parameter(parameter, 'Histogram_Window_Weight', false, 'rectangle' );  %Weighting for VEC and VEO are histogram processing. Type are Gaussian,Dual Rayleigh,Triangle, and Rectangle (default)
param.sigma_r=xls_parameter(parameter, 'sigma_r', true, .020 ); % sigma_r for 0.3ck Gaussian histogram window. Unit are UI. Preferred usage.
param.Qr=xls_parameter(parameter, 'Qr', true, param.sigma_r ); % sigma_r replaces Qr gasussian histogram window. Unit are UI
param.QL=xls_parameter(parameter, 'QL', true, param.T_O/param.Qr/1000 ); % superceded with sigma_r but is the internal values that is used

%%

param.skew_ps=xls_parameter(parameter, 'skew_ps', true, 0 );% experiment p/n skew. Not used.
param.imb_Z_fctr=xls_parameter(parameter, 'imb_Z_fctr', true, 1 ); % exprimental p/n impedance missmatch.  Not used.
param.imb_C_fctr=xls_parameter(parameter, 'imb_C_fctr', true, 1 ); % exprimental p/n capacitance missmatch.  Not used.
param.awgn_mv=param.AC_CM_RMS;
param.flip=xls_parameter(parameter, 'flip', true, 0 );  % exprimental p/n missmatch flip.  Not used.
param.f_hp=xls_parameter(parameter, 'f_hp', true, 0 );  % for rx testing for eq 162-12 if 0 (default) then rx test using rx bbn 
param.Q=xls_parameter(parameter, 'Q', true, 0 );  % Implementation penalty for MLSE in EQ 178a-36

%% Adding new parameters to reveal whether Floating DFE or Floating RXFFE is used
% This removes the dependency on checking param.N_bg (that is no longer valid to reveal if floating DFE is used)
param.Floating_RXFFE=false;
param.Floating_DFE=false;
if param.N_bg > 0
    param.Floating_DFE=true;
end
if OP.RxFFE
    param.Floating_DFE=false;
    if param.N_bg > 0
        param.Floating_RXFFE=true;
    end
end
%% for introducing Tx or Rx skew on p leg or n leg
param.Txpskew=xls_parameter(parameter, 'Txpskew', true, 0 );  % Tx p skew in ps
param.Txnskew=xls_parameter(parameter, 'Txnskew', true, 0 );  % Tx n skew in ps
param.Rxpskew=xls_parameter(parameter, 'Rxpskew', true, 0 );  % Rx p skew in ps
param.Rxnskew=xls_parameter(parameter, 'Rxnskew', true, 0 );  % Rx n skew in ps

%%
OP.TIMESTAMP= xls_parameter(parameter, 'TIMESTAMP', false,0); % Debug feature: a time stamp string is appended to saved file names
OP.WRITE_CSV_TRANSPOSED= xls_parameter(parameter, 'WRITE_CSV_TRANSPOSED', false,0); % Debug feature: Writes the csv file one field per row
OP.DO_NOT_COMPUTE_COM = xls_parameter(parameter, 'DO_NOT_COMPUTE_COM', false,0); % Switch to turn off optimize FOM/COM and report other computed parameters (ERL, ICN, etc) in output_args
OP.DO_NOT_COMPUTE_COM = xls_parameter(parameter, 'NO_COM', false, OP.DO_NOT_COMPUTE_COM); % Switch to turn off optimize FOM/COM and report other computed parameters (ERL, ICN, etc) in output_args same as as DO_NOT_COMPUTE_COM
OP.include_pcb = xls_parameter(parameter, 'Include PCB', false); % Used to add a PCB one each side of the passed s-parameters.
OP.exit_if_deployed = xls_parameter(parameter, 'exit if deployed', false,0); % may need set when COM is an exe
OP.INCLUDE_CTLE = xls_parameter(parameter, 'INCLUDE_CTLE', false, 1); % do not use 
OP.EXE_MODE= xls_parameter(parameter, 'EXE_MODE', false, 1);% 12/21 0:legacy 1:fast 2:superfast default is 1.
OP.INCLUDE_FILTER = xls_parameter(parameter, 'INCLUDE_TX_RX_FILTER', false, 1); % do not use
OP.force_pdf_bin_size = xls_parameter(parameter, 'Force PDF bin size', false, 0); % do not use
OP.BinSize = xls_parameter(parameter, 'PDF bin size', false, 1e-5); % set lower for faster computation time but less accuracy. 
OP.DEBUG = xls_parameter(parameter, 'DIAGNOSTICS', false, false); % supresss some interim compuation value printouts
OP.DISPLAY_WINDOW = xls_parameter(parameter, 'DISPLAY_WINDOW', false, true); % controls if graph plots are displayed. Typically goes along with DIAGNOSTICS
OP.CSV_REPORT = xls_parameter(parameter, 'CSV_REPORT', false, true); % saves all the output parameters to a CSV file in the results directory, If DIAGNOSTICS is set then a mat file is also created
OP.SAVE_TD=xls_parameter(parameter, 'SAVE_TD', false, false); % Save the time domian waveforms. FIR, PR etc. in an output structure
OP.SAVE_FIGURES=xls_parameter(parameter, 'SAVE_FIGURES', false, false); % save displayed figures in the results directory
OP.SAVE_FIGURE_to_CSV=xls_parameter(parameter, 'SAVE_FIGURE_to_CSV', false, false); % does not work. do not use.
OP.GET_FD = xls_parameter(parameter, 'Display frequency domain', false, OP.GET_FD); % Not normally set in the config file. It is normally just set to true to get FD plots
OP.INC_PACKAGE = xls_parameter(parameter, 'INC_PACKAGE', false, true); % warning: INC_PACKAGE=0 not fully supported, instead, set Zp,Cd, and Cp parameters to zero and Zp select to 1
if ~OP.INC_PACKAGE
    fprintf('<strong> Warning!!! INC_PACKAGE=0 not fully supported, instead, set Zp,Cd, and Cp parameters to zero and Zp select to 1 </strong>\n');
end

OP.EW = xls_parameter(parameter, 'EW', false, false); % RIM 3-18-2021 change defaults
OP.IDEAL_TX_TERM = xls_parameter(parameter, 'IDEAL_TX_TERM', false, false);
if OP.IDEAL_TX_TERM
    fprintf('<strong> Warning!!! IDEAL_TX_TERM not supported, instead, set Zp,Cd, and Cp parameters to zero and Zp select to 1 </strong>\n');
end
OP.IDEAL_RX_TERM = xls_parameter(parameter, 'IDEAL_RX_TERM', false, false);
if OP.IDEAL_RX_TERM
    fprintf('<strong> Warning!!! IDEAL_RX_TERM not supported, instead, set Zp,Cd, and Cp parameters to zero and Zp select to 1 </strong>\n');
end

OP.TDMODE = xls_parameter(parameter, 'TDMODE',false, OP.TDMODE);   % Enables the the use of pulse response instead of s-parameters. Assumes no packages or the packages are included in the PR. Default is 0.

OP.FT_COOP = xls_parameter(parameter, 'FT_COOP',false, false); % obsolete do not use.
OP.RESULT_DIR = regexprep(xls_parameter(parameter, 'RESULT_DIR'), '\\', filesep); % directory where results like csv, mat, and/or figure files will be written
OP.RESULT_DIR=strrep(OP.RESULT_DIR,'{date}',date);
OP.BREAD_CRUMBS = xls_parameter(parameter, 'BREAD_CRUMBS',false, false); % if DIAGNOSTICS is set then param, OP, and chdata are include in the output for each run
OP.BREAD_CRUMBS_FIELDS = xls_parameter(parameter, 'BREAD_CRUMBS_FIELDS',false, ''); % if BREAD_CRUMBs is enabled, this file controls what chdata fields are included
OP.COM_CONTRIBUTION_CURVES = xls_parameter(parameter, 'COM_CONTRIBUTION',false,0);   % Default is 0. If set to 1 then a bar graph of COM contributors is produce instead of bathtub curves
OP.ENFORCE_CAUSALITY = xls_parameter(parameter, 'Enforce Causality', false, 0);% default is 0. Not recommended
OP.EC_REL_TOL = xls_parameter(parameter, 'Enforce Causality REL_TOL', false, 1e-2); % Relative Tolerance parameter for causality, Hard enforcement, 1e-3, Soft enforcement,  1e-2
OP.EC_DIFF_TOL = xls_parameter(parameter, 'Enforce Causality DIFF_TOL', false, 1e-3); % Difference Tolerance parameter for causality, Hard enforcement, 1e-4,Soft enforcement, 1e-3
OP.EC_PULSE_TOL = xls_parameter(parameter, 'Enforce Causality pulse start tolerance', false, 0.01); % Tolerance parameter for causality, Hard enforcement, 0.05, Soft enforcement, .01
OP.pkg_len_select = xls_parameter(parameter, 'z_p select', true, 1);  % List of package length indexes used to run COM
% rx calibration
OP.RX_CALIBRATION = xls_parameter(parameter, 'RX_CALIBRATION', false, false); % Turn on RX_Calibration loop for non MMSE
OP.PSDRXCAL = xls_parameter(parameter, 'PSDRXCAL', false, false); % Turn on RX_Calibration loop with PSD for MMSE
OP.RIT_REF_PTR= xls_parameter(parameter, 'RIT_REF_PTR', false, 'clause_179'); % either 'clause_179' (d) or 'clause_178'
OP.sigma_bn_STEP = xls_parameter(parameter, 'Sigma BBN step', false, 5e-3); % BBN step for Rx Calibration in volts. Defaults is 0.5e-3
OP.BBN_Q_factor = xls_parameter(parameter, 'BBN Q factor', false, 5);   % Overrides NEXT/FEXT noise Qfactor for  'Force BBN Q factor' used for reporting. does not affect COM.
OP.force_BBN_Q_factor = xls_parameter(parameter, 'Force BBN Q factor', false, false); % Used for reporting and bathtub curves. does not affect COM.
OP.transmitter_transition_time = xls_parameter(parameter, 'T_r', true , 8e-3); % 20% to 80% transition time used for the Gaussian shaped source
OP.RL_norm_test=xls_parameter(parameter, 'ERL_FOM', false, 1); % Defaults to 1 indicating variance is used for FOM determination.  Do not change.

OP.T_r_meas_point = xls_parameter(parameter, 'T_r_meas_point', false, 0); % included for earlier version support. Not recommended to use.
OP.T_r_filter_type= xls_parameter(parameter, 'T_r_filter_type', false, 0);% included for earlier version support. Not recommended to use.
OP.FORCE_TR = xls_parameter(parameter, 'FORCE_TR', false, false);% Included for earlier version support but should be set to 1 in most later config sheets.
% Control with OP.T_r_filter_type and OP.T_r_meas_point for backward
% compatibility
if OP.FORCE_TR
    OP.T_r_meas_point=0;
    OP.T_r_filter_type=1;
end
OP.TDR = xls_parameter(parameter, 'TDR', false, false); % Set to 1 to produce TDR results
OP.TDR_duration= xls_parameter(parameter, 'TDR_duration', false, 5); % only used if N*UI is longer than the TDR duration time.  Default is 5 times the raw s-parameter transit time.
OP.N = xls_parameter(parameter, 'N', false, 0); % duration time in UI which is used for ERL (PTDR)
OP.WC_PORTZ = xls_parameter(parameter, 'WC_PORTZ', false, false); % Do not use: Obsolete. 
OP.T_k= xls_parameter(parameter, 'T_k', false, .6)*1e-9; % Time span (ns) for which the impedance of port is determined using TDR.
OP.ERL_ONLY = xls_parameter(parameter, 'ERL_ONLY', false,0); % Compute ERL only
OP.ERL=xls_parameter(parameter, 'ERL', false, false); % Enables ERL. Needs TDR to be set as well.
if OP.ERL
    OP.PTDR=1;
else
    OP.PTDR=0;
end % ERL needs to do a TDR
OP.Report_Modal_ERL=xls_parameter(parameter, 'Report_Modal_ERL', false, '0'); % Enables reporting of modal ERL. default = 0; values={ENABLE,PROVISIONAL}
OP.SHOW_BRD= xls_parameter(parameter, 'SHOW_BRD', false,0);% include added board (PCB) in TDR and ERL. Default is 0.
if OP.WC_PORTZ , OP.TDR=1;end % Obsolete: WC_PORTZ needs to do a TDR
OP.TDR_W_TXPKG = xls_parameter(parameter, 'TDR_W_TXPKG', false,0);% adds tx package for TDR, PTDR, and ERL. Default is 0.
OP.Bessel_Thomson=xls_parameter(parameter, 'Bessel_Thomson', false, false); % enable Bessel Thomsen filter for COM
OP.TDR_Butterworth=xls_parameter(parameter, 'TDR_Butterworth', false, true); % enable Butterworth filter for TDR, PTDR, and ERL
OP.Butterworth=xls_parameter(parameter, 'Butterworth', false, 1); % Enable Butterworth Rx filter for COM compuatetopm
OP.Raised_Cosine=xls_parameter(parameter, 'Raised_Cosine', false,0); % Not used if 0. Default is zero. Should set BT and BW to false
OP.inc_reflect_board=xls_parameter(parameter, 'inc_reflect_board', false,0); % Not used if 0. Default is zero.
OP.AUTO_TFX=xls_parameter(parameter, 'AUTO_TFX', false,0); % Mostly used for device ERL. If sent to 1 the fixture tfx will be estimated.
OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN = xls_parameter(parameter, 'LIMIT_JITTER_CONTRIB_TO_DFE_SPAN', false, false); % Experimental. Default is 0.
%OP.impulse_response_truncation_threshold = xls_parameter(parameter, 'Impulse response truncatio threshold', false, 1e-3);
OP.impulse_response_truncation_threshold = xls_parameter(parameter, 'Impulse response truncation threshold', false, 1e-3); % zero padding threshold in fraction of IR peak for the impulse response. Effectively controls the length of time for the PR. Larger values decrease run time and accuracy. Default is 1e-3.
OP.interp_sparam_mag = xls_parameter(parameter, 'S-parameter magnitude extrapolation policy', false, 'linear_trend_to_DC'); % magnitued extrapolation method
OP.interp_sparam_phase = xls_parameter(parameter, 'S-parameter phase extrapolation policy', false, 'extrap_cubic_to_dc_linear_to_inf'); % phase extrapolation method
OP.ZERO_PAD= xls_parameter(parameter, 'ZERO_PAD', false, 0); % phase extrapolation method
OP.PMD_type= xls_parameter(parameter, 'PMD_type', false,'C2C'); %  Either C2C or C2M. C2M is for computing VEC and VEO
OP.PHY= xls_parameter(parameter, 'PHY', false, OP.PMD_type); % The keyword OP.PMD_type is now used
if strcmpi(OP.PHY,'C2M') 
    OP.EW=true;
else
    param.T_O=0; % make sure when c2c that sample is at Ts
end
if param.Min_VEO ~=0 && strcmpi(OP.PHY,'C2C')
    OP.PHY='C2Mcom';
end
OP.TDECQ=xls_parameter(parameter, 'TDECQ', false, 0); % Experimental, for only option is none (0) or vma. Default is 0.
switch lower(OP.TDECQ)
    case {false 'none' 'vma'}
    otherwise
        error('%s unrecognized TDECQ keyword',OP.TDECQ)
end
OP.RUNTAG = xls_parameter(parameter, 'RUNTAG', false, ''); % This string is appended to the begining of results files 
if isnan(OP.RUNTAG), OP.RUNTAG='';end
if isnumeric(OP.RUNTAG), OP.RUNTAG=num2str(OP.RUNTAG);end
OP.CDR=xls_parameter(parameter, 'CDR', false, 'MM');% 12/21 from Yuchun Lu to accomdate 'Mod-MM', Defautt is 'MM'
OP.Optimize_loop_speed_up =xls_parameter(parameter, 'Optimize_loop_speed_up', true , 0);% If set to 0 (or default) normal looping, If set to 1 loop speedup by slightly reducing PD Fbin and FIR_threshold for optimize looping only  
% Parameters for error burst probability calculation. Not officially used
OP.use_simple_EP_model = xls_parameter(parameter, 'Use simple error propagation model', false, false);% Use to calculate burst error rate (not normally used
OP.nburst = xls_parameter(parameter, 'Max burst length calculated', false, 0); % Use to calculate burst error rate (not normally used)
OP.COM_EP_margin = xls_parameter(parameter, 'Error propagation COM margin', false, 0); % Use to calculate  error propogation (not normally used)
OP.USE_ETA0_PSD = xls_parameter(parameter, 'USE_ETA0_PSD', false, 0); % Used eta_0 PSD equaiton for sigma_n. Default is 0. Do not use.
OP.SAVE_CONFIG2MAT = xls_parameter(parameter, 'SAVE_CONFIG2MAT', false, 0); % If set to 1 (default) saves parameters in mat file. Requires DIAGNOSTICS to be set.
OP.PLOT_CM = xls_parameter(parameter, 'PLOT_CM', false, 0); % plots all modal reponces if set to 1. Default is 0.
OP.CM_MASK_REPORT =  xls_parameter(parameter, 'CM_MASK_REPORT', false, 0); % Display CM plots if set to 1. Default is 0.
OP.fraction_of_F_range_start_extrap_from= xls_parameter(parameter, 'fraction_of_F_range_start_extrap_from', true, 0.75); % Frequency (fb) where high frequency extropolation begins for computing IR. Helps control Gibbs phenomena. defualt is 0.75.
OP.COMPUTE_RILN = xls_parameter(parameter, 'COMPUTE_RILN', false, 0); % Computes RILN default is 0.  FOM_RILN reported
OP.COMPUTE_TDILN = xls_parameter(parameter, 'COMPUTE_TDILN', false, OP.COMPUTE_RILN); %  computes TD ILN from complex freq IL fit. FOM_TDILN reported.
OP.SAVE_KEYWORD_FILE = xls_parameter(parameter, 'SAVE_KEYWORD_FILE', false, 0); % Save csv file of COM parameter (OP) and keywords. Not implemented.
OP.SNR_TXwC0 = xls_parameter(parameter, 'SNR_TXwC0', false, 0); % Adjust SNR_TX with C0
OP.MLSD = xls_parameter(parameter, 'MLSD', false, 0); % not 0 or missing means use MLSD
OP.MLSE = xls_parameter(parameter, 'MLSE', false, OP.MLSD ); % MSLD Synonym support
if OP.MLSE ~=0
    if param.T_O ~= 0
        error('MLSD nnot presently no supported for VEC')
    end
    if OP.COM_CONTRIBUTION_CURVES ~=0
        warning('COM_CONTRIBUTION_CURVES not functional yet with MLSE')
        OP.COM_CONTRIBUTION_CURVES=0;
    end
end
OP.RXFFE_FLOAT_CTL =  xls_parameter(parameter, 'RXFFE FLOAT CTL', false, 'FOM'); % select taps (taps), pulse response (ISI), or FOM for floating taps determination
OP.RXFFE_TAP_CONSTRAINT =xls_parameter(parameter, 'RXFFE TAP CONSTRAINT', false, 'Unity Cursor'); % "Unity sum taps", "Unity Cursor", of unbounded
if OP.MLSE && param.ndfe==0
        error('At least DFE 1 must be set to use MLSE');
end
OP.TIME_AXIS = xls_parameter(parameter, 'TIME_AXIS', false, 'UI'); % if0 OP.display set pulse response xaxis to seconds or UI
% MNSE parameters
OP.Do_XT_Noise= xls_parameter(parameter, 'Do_XT_Noise', false, 1);
OP.FFE_SNR= xls_parameter(parameter, 'FFE_SNR', false, 1);
OP.Do_Colored_Noise= xls_parameter(parameter, 'Do_Colored_Noise', false, 1);
OP.Do_White_Noise=xls_parameter(parameter, 'Do_White_Noise', false, 0);
OP.FFE_OPT_METHOD=xls_parameter(parameter,'FFE_OPT_METHOD',false,'MMSE'); % 'MMSE','FV-LMS'
% Commit request 4p4_7, healey_3dj_COM_01_240416
OP.TS_SRCH_MODE=xls_parameter(parameter,'TS_SRCH_MODE',false,'full-sweep'); % full-sweep, middle 
% need to make sure TD mode does not invoke FD operations
if OP.TDMODE % need to set GET_FD false of TDMODE
    OP.GET_FD=false;
    OP.ERL_ONLY=0;
    OP.ERL=0;
    OP.PTDR=0;
    OP.TDR=0;
    OP.RX_CALIBRATION=0;
    OP.PSDRXCAL = 0;
end

% If the user has set RX Calibration with RxFFE and MMSE, then swap to the new PSDRXCAL parameter
if OP.RX_CALIBRATION
    if strcmp(OP.FFE_OPT_METHOD,'MMSE') && OP.RxFFE
        OP.PSDRXCAL = 1;
        OP.RX_CALIBRATION = 0;
    end
end

% Trigger some errors on bad parameter combinations
if OP.PSDRXCAL
    if OP.RX_CALIBRATION
        error('RX Calibration must be turned off when PSDRXCAL is enabled');
    end
    if ~strcmp(OP.FFE_OPT_METHOD,'MMSE')
        error('FFE Method must be "MMSE" when PSDRXCAL is enabled');
    end
    if ~OP.RxFFE
        error('RxFFE must be enabled when PSDRXCAL is enabled');
    end
end

if OP.SAVE_CONFIG2MAT || OP.CONFIG2MAT_ONLY
    save(matcongfile ,'parameter');
end

OP.SNDR_REF= xls_parameter(parameter,'SNDR_REF',false,0); % controls whether SNDR_REF is computeted. Heed special config sheet
preset_txffe=xls_parameter(parameter,'preset_txffe',true,[]); % controls whether SNDR_REF is computeted. Heed special config sheet
if ~isempty(preset_txffe)
    for ii=1:length(preset_txffe),param.preset(ii).txffe=preset_txffe(ii,:);end
end

%% At the very end of Parameter reading, swap in the proper Tx and Rx values for package parameters based on pkg name
if ~isempty(param.PKG_NAME)
    if length(param.PKG_NAME) == 1
        param.PKG_NAME = [param.PKG_NAME param.PKG_NAME];
    end
    tx_rx_fields = {'C_pkg_board' 'R_diepad'};
    tx_rx_fields_matrix = {'pkg_Z_c'};
    tx_fields = {'z_p_tx_cases' 'z_p_fext_cases' 'pkg_gamma0_a1_a2' 'pkg_tau' 'a_thru' 'a_fext'};
    rx_fields = {'z_p_rx_cases' 'a_next' 'z_p_next_cases'};
    tx_pkg_name=param.PKG_NAME{1};
    rx_pkg_name=param.PKG_NAME{2};
    tx_pkg_struct=param.PKG.(tx_pkg_name);
    rx_pkg_struct=param.PKG.(rx_pkg_name);
    
    %tx_rx_fields: put the value from the tx package in the Tx position and the value from the rx package in the RX position
    for j=1:length(tx_rx_fields)
        tx_val = tx_pkg_struct.(tx_rx_fields{j});
        rx_val = rx_pkg_struct.(tx_rx_fields{j});
        param.(tx_rx_fields{j}) = [tx_val(1) rx_val(2)];
    end
    
    %tx_rx_fields_matrix:  same as tx_rx_fields but in matrix form
    for j=1:length(tx_rx_fields_matrix)
        tx_val = tx_pkg_struct.(tx_rx_fields_matrix{j})(1,:);
        rx_val = rx_pkg_struct.(tx_rx_fields_matrix{j})(2,:);
        param.(tx_rx_fields_matrix{j}) = [tx_val; rx_val];
    end
    
    %tx_fields:  use only the tx package values
    for j=1:length(tx_fields)
        param.(tx_fields{j}) = tx_pkg_struct.(tx_fields{j});
    end
    
    %rx_fields:  use only the rx package values
    for j=1:length(rx_fields)
        param.(rx_fields{j}) = rx_pkg_struct.(rx_fields{j});
    end

end


%%
