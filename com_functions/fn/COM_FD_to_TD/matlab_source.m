function chdata=COM_FD_to_TD(chdata,param,OP)
% get impulse responses which in interim step between equation for X(f) and
% H^(k)(t) without TX FFE or CTLE. These will we added later.
case_number=param.package_testcase_i;
M=param.samples_per_ui;
%% power functions RIM 04/14/2025
Sinc=@(x) sin(pi*x+eps(0))./(pi*x+eps(0));
W = @(f,ft,fr,fb)            1/fb*    Sinc(f/fb).^2.* (1./(1+(f/ft).^4)) .* (1./(1+(f/fr).^8)); % eq 93A-57
H_bt=Bessel_Thomson_Filter(param,chdata(1).faxis,OP.Bessel_Thomson);
H_bw=Butterworth_Filter(param,chdata(1).faxis,OP.Butterworth);
H_r=H_bw.*H_bt;
H_t = exp(-(pi*chdata(1).faxis/1e9*OP.transmitter_transition_time/1.6832).^2);
H_filters=H_r.*H_t;
for i=1:param.number_of_s4p_files
% chdata(i).sdd21 is the voltage transfer function 
% amplitude chdata(i).A is add in this function
% and filtered by fr in FD_Processing.m and also has packages and added
% boards (if specified in the config sheet)
% s21_pkg is where the vtf and tr filter are added 
% chdata.sdd21_orig is the passed sdd21 and used for SCMR 
% for chdata.sdd21_raw tr and fr filter are added 
% ss =@(a) sum(abs(a(1:length(a))).^2);
% prior version did no have filter applied when computing SCMR
% uneq_imp_response, uneq_imp_response_raw,  uneq_imp_response_raw may only be needed fror i=1
    [chdata(i).uneq_imp_response, ...
        chdata(i).t, ...
        chdata(i).causality_correction_dB, ...
        chdata(i).truncation_dB] = s21_to_impulse_DC(chdata(i).sdd21 ,chdata(i).faxis, param.sample_dt, OP,param) ; %  w/ package/boards and filters
    chdata(i).uneq_pulse_response=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_imp_response);
    [chdata(i).uneq_imp_response_raw, ...
        chdata(i).t_raw, ...
        chdata(i).causality_correction_dB, ...
        chdata(i).truncation_dB] = s21_to_impulse_DC(chdata(i).sdd21_raw ,chdata(i).faxis, param.sample_dt, OP,param) ;  % just passed s-parameters    
    chdata(i).uneq_pulse_response_raw=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_imp_response_raw);
    % apply Tr and fr filters
    [chdata(i).uneq_imp_response_raw_filtered, ...
        chdata(i).t_raw_fltr, ...
        chdata(i).causality_correction_dB, ...
        chdata(i).truncation_dB] = s21_to_impulse_DC(chdata(i).sdd21_raw.*H_filters ,chdata(i).faxis, param.sample_dt, OP,param) ;  % just passed s-parameters    
    % need unequalize PR of orig for SCMR_CH
     [chdata(i).uneq_imp_response_orig, ...
        chdata(i).t_orig, ...
        chdata(i).causality_correction_dB, ...
        chdata(i).truncation_dB] = s21_to_impulse_DC(chdata(i).sdd21_orig ,chdata(i).faxis, param.sample_dt, OP,param) ;  % just passed s-parameters    
    chdata(i).uneq_pulse_response_orig=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_imp_response_orig);
    chdata(i).uneq_pulse_response_orig_filtered=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_pulse_response_orig);
        [chdata(i).uneq_imp_response_orig_filtered, ...
        chdata(i).t_orig_fltr, ...
        chdata(i).causality_correction_dB, ...
        chdata(i).truncation_dB] = s21_to_impulse_DC(chdata(i).sdd21_orig.*H_filters ,chdata(i).faxis, param.sample_dt, OP,param) ;  % just passed s-parameters    
    chdata(i).uneq_pulse_response_orig_filtered=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_imp_response_orig_filtered);

    %
    % check conditions to use channel amplitude and calculate common to differential
    USE_channel_amplitude = 1;
    if OP.RX_CALIBRATION && i>1
        USE_channel_amplitude = 0;
    end
    if OP.PSDRXCAL && i == length(chdata)
        USE_channel_amplitude = 0;
    end
    if USE_channel_amplitude
        chdata(i).uneq_imp_response=chdata(i).uneq_imp_response*chdata(i).A; % adjust IRx for amplitude
    end
    [chdata(i).uneq_CD_imp_response_filtered, ...
        chdata(i).t_CD_fltr, ...
        chdata(i).causality_correction_CD_dB, ...
        chdata(i).truncation__CD_dB] = s21_to_impulse_DC(chdata(i).scd21_orig.*H_filters ,chdata(i).faxis, param.sample_dt, OP,param) ; % just passed s-parameters
    chdata(i).uneq_pulse_CD_response_filtered=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_CD_imp_response_filtered);
    % DC is using raw passed s-parameters
    % [chdata(i).uneq_DC_imp_response, ...
    %     chdata(i).t_DC, ...
    %     chdata(i).causality_correction_DC_dB, ...
    %     chdata(i).truncation__CD_dB] = s21_to_impulse_DC(chdata(i).sdc21_orig ,chdata(i).faxis, param.sample_dt, OP,param) ; % just passed s-parameters
    % chdata(i).uneq_pulse_DC_response_raw=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_DC_imp_response);
    [chdata(i).uneq_DC_imp_response_filtered, ...
        chdata(i).t_DC_fltr, ...
        chdata(i).causality_correction_DC_dB, ...
        chdata(i).truncation__DC_dB] = s21_to_impulse_DC(chdata(i).sdc21_orig.*H_filters ,chdata(i).faxis, param.sample_dt, OP,param) ; % just passed s-parameters
    chdata(i).uneq_pulse_DC_response_filtered=filter(ones(1, param.samples_per_ui), 1, chdata(i).uneq_DC_imp_response_filtered);
    %------------------------------------------------------------
    % next find Pulse response (SBR) for each channel h^(k)(t)
    if ~OP.DISPLAY_WINDOW && i==1, fprintf('processing COM PDF '); end
    % differtial to common mode conversion (includes channel skew)
    if 1 % also in FD_Processing... could combine latter not used right. will be used to compare FD to TD power ratios
        a=find(chdata(i).faxis(:)>=param.f2,1,'first');% RIM 01-12-21
        if isempty(a)
            index_f2=length(chdata(i).faxis);
        else
            index_f2=a(1);
        end
        b=find(chdata(i).faxis(:)<=param.f1,1,'last');% RIM 01-12-21
        if isempty(b)
            index_f1=1;
        else
            index_f1=b(1);
        end
    end
    %
    rss=-inf; % Needs work on commom mode noise
    for im=1:param.samples_per_ui
        rss=max(rss, norm( chdata(i).uneq_pulse_CD_response_filtered(im:param.samples_per_ui:end)));
    end
    chdata(i).CD_CM_RMS=rss*sqrt(param.sigma_X);  % not vetted
    chdata(i).VCM_CD_HF_struct= get_cm_noise(param.samples_per_ui,chdata(i).uneq_pulse_CD_response_filtered,param.levels,param.P_peak,OP); % returns voltages % changed for d 2.2 from param.P_peak
    chdata(i).VCM_DC_HF_struct= get_cm_noise(param.samples_per_ui,chdata(i).uneq_pulse_DC_response_filtered,param.levels,param.P_peak,OP); % returns voltages % changed for d 2.2 from param.P_peak
    % find the peak of the uneq_pulse_response_raw
    PR_ORIG_fltr=chdata(1).uneq_pulse_response_orig_filtered; % the signal is defined only for the through channel
    ipeak=find(PR_ORIG_fltr==max(PR_ORIG_fltr),1,'first'); %#ok<NASGU>
    V_peak=PR_ORIG_fltr(ipeak);
    istart=mod(ipeak-1,M)+1;
    iend=floor((length(PR_ORIG_fltr)/M))*M;
    PR_ORIG_fltr_sampled=PR_ORIG_fltr(istart:M:iend);
    P_signal=norm(PR_ORIG_fltr_sampled)^2;
    sigma_ts=norm(PR_ORIG_fltr_sampled);
    chdata(i).P_signal=P_signal; % all files use the same P_signal
    chdata(i).SCMR_CD_ch_pk=10*log10(V_peak^2/chdata(i).VCM_CD_HF_struct.CMn^2);
    chdata(i).SCMR_CD_ch=10*log10(P_signal/chdata(i).VCM_CD_HF_struct.CMn^2); % power after filters
    chdata(i).SCMR_DC_ch_pk=10*log10(V_peak^2/chdata(i).VCM_DC_HF_struct.CMn^2);
    chdata(i).SCMR_DC_ch=10*log10(P_signal/chdata(i).VCM_DC_HF_struct.CMn^2); % power after filters
    if OP.DEBUG && OP.DISPLAY_WINDOW
        if OP.DISPLAY_WINDOW && ~OP.RX_CALIBRATION
            figure(150+case_number);set(gcf,'Tag','COM');
            screen_size=get(0,'ScreenSize');
            pos = get(gcf, 'OuterPosition');
            set(gcf, 'OuterPosition', ...
                screen_size([3 4 3 4]).*[1 1 0 0] + pos([3 4 3 4]).*[-1 -1 1 1] ...
                - (case_number-1)*[0 20 0 0]);
            %movegui(gcf,'northeast')
            set(gcf, 'Name', sprintf('Case %d PR & PDF - %s', case_number, chdata(i).base));
            subplot(2,1,1);  hold on; % all plots on the same axes
            hp=plot(chdata(i).t_raw, chdata(i).uneq_pulse_response_raw,'Disp', chdata(i).base);
            hold on; % leave on for s-parameter problem finding. RIM 10-02-2023
            hp1=plot(chdata(i).t_CD_fltr, chdata(i).uneq_pulse_CD_response_filtered,'Disp', [ 'CD CM ' chdata(i).base ]) ;
            hp2=plot(chdata(i).t_DC_fltr, chdata(i).uneq_pulse_DC_response_filtered,'Disp', [ 'DC CM'  chdata(i).base ]) ;
        end
        % hide thru PR in order to show xtalk in a reasonable
        % scale. thru is shown in another plot.
        if isequal(chdata(i).type, 'THRU' ) && ~OP.RX_CALIBRATION %|| OP.RX_CALIBRATION % RIM 06-14-2022
            % set(hp, 'visible', 'off');
            % set(get(get(hp,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');
        end
        title(sprintf('Unequalized Crosstalk and CD Conversion \n Pulse Responses'))
        ylabel('Volts')
        xlabel('seconds')
        recolor_plots(gca);
    else
        if param.ndfe~=0
            fprintf('%s\tUnequalized pulse peak = %.1f mV\n', chdata(i).base, 1000*max(abs(chdata(i).uneq_pulse_response)));
        end
    end

    fprintf('%s\tCausality correction = %.1f dB', chdata(i).base, chdata(i).causality_correction_dB);
    if OP.ENFORCE_CAUSALITY
        fprintf('\n');
    else
        fprintf(' (not applied)\n');
    end
    fprintf('%s\tTruncation ratio = %.1f dB\n', chdata(i).base, chdata(i).truncation_dB);

end
