function result = get_PSDs(result,h,cursor_i, txffe,G_DC,G_DC2,param,chdata,OP)
% OP.COMPUTE_COM is when called after the optimization and returns sigma_Gbest_hk
% as well the Sn with the Rx ffe and Hisi included as sigma ISI inclued
if 1 % force indent for doc
    num_ui=param.num_ui_RXFF_noise;
    M=param.samples_per_ui;
    L=param.levels;
    f_b=param.fb;
    SNR_TX=param.SNR_TX;
    dw=param.RxFFE_cmx;
    bmax=param.bmax;
    bmin=param.bmin ;
    Nb=param.ndfe;
    sigma_X2=(L^2-1)/(3*(L-1)^2);
    eta_0=param.eta_0; %V^2/GHz
    T_b=1/f_b;
    delta_f = f_b/num_ui; % Units are Hz.
    fvec = (0:num_ui*M/2)*delta_f; % Single-sided frequency axis.
    result.fvec=fvec;
end
if OP.COMPUTE_COM
    %% H_rxffe eq 178A-28 d1.0
    % chdata(xchan).ctle_imp_response is conputed with the RxFFE during the COM compuatation
    % H_rxffe.^2 is distributed S_jn and S_rn since they do not use use chdata(xchan).ctle_imp_response
    H_rxffe=0;
    for nn=1:length(result.w)
        H_rxffe=result.w(nn)*exp(-1j*2*pi*fvec*T_b*(nn-dw-1))+H_rxffe;
    end
    H_rxffe_2_of_f=abs(H_rxffe).^2;
    H_rxffe_2=H_rxffe_2_of_f(1:num_ui/2+1);
    H_rxffe_2= [real( H_rxffe_2(1)),  H_rxffe_2(2:end-1), real( H_rxffe_2(end)), conj( H_rxffe_2(end-1:-1:2))];
else
    H_rxffe_2=1;
end
result.H_rxffe_2=H_rxffe_2;% pass as output for compuation in healey_3dj_01_2409 slide 12
if OP.WO_TXFFE % to speed up loop find sn onlu first time ctle is case
    % --->this is the point in the code may fork where we add extra rx noise
    %% compute S_rn ( eq 178A-15 d0.2 )
    if ~OP.COMPUTE_COM
        S_RN_of_f=S_RN(fvec,G_DC,G_DC2,param);
        rxn_psd=[real(S_RN_of_f(1)), S_RN_of_f(2:end-1), real(S_RN_of_f(end)), conj(S_RN_of_f(end-1:-1:2))]; % Convert single-sided frequency response to conjugate-symmetric
        rxn_psd=rxn_psd/1e9;% Units are V^2/Hz.
        rxn_rms = sqrt(sum(rxn_psd)* delta_f);
        S_rn = sum(reshape(rxn_psd, num_ui, M).');
        S_rn=S_rn(1:num_ui/2+1);
        S_rn= [real( S_rn(1)),  S_rn(2:end-1), real( S_rn(end)), conj( S_rn(end-1:-1:2))];
        result.S_rn=S_rn;
        result.S_rn_rms = sqrt(sum(result.S_rn)* delta_f);
        
        %% compute S_in ( eq 178A-24 d0.2 )
        % S_in = 0 when Rx Calibration is not performed
        if OP.PSDRXCAL
            % interpolate to fvec
            result.H_noise=H_interp(chdata(end).sdd21p,chdata(1).faxis,fvec,f_b);
            % Units are V^2/Hz
            S_IN_of_f=S_IN(fvec,result.H_noise,G_DC,G_DC2,param,OP);
            % Convert single-sided frequency response to conjugate-symmetric
            inn_psd=[real(S_IN_of_f(1)), S_IN_of_f(2:end-1), real(S_IN_of_f(end)), conj(S_IN_of_f(end-1:-1:2))];
            inn_rms = sqrt(sum(inn_psd)* delta_f);
            S_in = sum(reshape(inn_psd, num_ui, M).');
            S_in=S_in(1:num_ui/2+1);
            S_in= [real( S_in(1)),  S_in(2:end-1), real( S_in(end)), conj( S_in(end-1:-1:2))];
            result.S_in=S_in;
            result.S_in_rms = sqrt(sum(result.S_in)* delta_f);
        else
            result.S_in=0;
            result.S_in_rms =0;
        end
    else
           %% compute S_rn
           result.S_rn=result.S_rn.*H_rxffe_2;
           result.S_rn_rms = sqrt(sum(result.S_rn)* delta_f);
           %% compute S_in
           result.S_in=result.S_in.*H_rxffe_2;
           result.S_in_rms = sqrt(sum(result.S_in)* delta_f);
    end

else % find noise for item that set have tx ffe for each loop
    %% S_xn from eq 178A-16
    %% Crosstalk power spectral density
    if ~OP.COMPUTE_COM % result.S_xn and result.S_xn_rms were found in optimizes_fom and passed in with the variable result
        result.S_xn=0;
        if OP.PSDRXCAL
            num_channel_files = length(chdata) - 1;
        else
            num_channel_files = length(chdata);
        end
        if num_channel_files~=1
            for xchan=2:num_channel_files
                pulse_ctle=filter(ones(1,M),1,chdata(xchan).ctle_imp_response(:).');
                pulse_ctle=[ pulse_ctle(1:floor(length(pulse_ctle)/M)*M) ];
                hk(xchan).k=chdata(xchan).pulse_response_w_CFT_TXFFE_noRxFFE.';
                % enable less UI for computation speed improvement
                %%
                if num_ui*M > length(pulse_ctle)
                    hk(xchan).k= [ hk(xchan).k zeros(1,num_ui*M-length(hk(xchan).k)) ];% crosstalk pulse responces
                else
                    hk(xchan).k=hk(xchan).k(1:num_ui*M);
                end
                for i1=1:M
                    hxn(i1)=norm(hk(xchan).k(i1:M:length(hk(xchan).k)) );
                end
                [~, iphase(xchan)] = max(hxn); % Return max index, applicable to vector of non-zero and zero values
                hk(xchan).hrn= hk(xchan).k(iphase(xchan):M:length(hk(xchan).k));
                result.hk(xchan).hrn= hk(xchan).hrn;
                hk(xchan).S_xn=sigma_X2*(abs(fft(hk(xchan).hrn))).^2/param.fb;
                result.S_xn=hk(xchan).S_xn+result.S_xn;
            end
            result.S_xn=result.S_xn;
            result.hk=hk;
            result.iphase=iphase;
            result.S_xn_rms = sqrt(sum(result.S_xn)* delta_f);
        else % if no crosstalk, perserve structure and return 0 for S_xn
            result.S_xn=0;
            result.hk=[];
            result.iphase=1;
            result.S_xn_rms = 0;
        end
    else % adjust for H_rxffe when computing COM
        result.S_xn=result.S_xn.*H_rxffe_2;
        result.S_xn_rms = sqrt(sum(result.S_xn)* delta_f);
    end
    %% S_tn from eq 178A-17
    %% if not in the opimization use value found in optimize_fom times |Hrxffe|^2
    %% Transmitter noise power spectral density
    if ~OP.COMPUTE_COM % "if" to "end" section changed by Hossein Shakiba to implement commit request 4p9_1
        if ~OP.TDMODE
            htn=filter(ones(1,M),1,chdata(1).ctle_imp_response); % ctle_imp_response does not have TxFFE included
        else % only use when the input was a pulse response not s-parameters
            if isfield(chdata(1),'ctle_pulse_response')
                htn=chdata(1).ctle_pulse_response;
            else
                htn=filter(ones(1,param.samples_per_ui),1, chdata(1).ctle_imp_response);
            end
        end
        
        % align to sample point
        htn=htn(mod(cursor_i-1,M)+1:end);
        htn=reshape(htn,1,[]); % make row vectors
        htn=htn(1:M:end);% resample
        len_htn = length(htn);
        if num_ui>len_htn
            hext=[htn zeros(1,num_ui-len_htn)];
        else
            hext=htn(1:num_ui);
        end
        
        % result.S_tn=sigma_X2*10^(-SNR_TX/10)*(abs(fft(hext))).^2/param.fb; % this corresponds to +/- pi
        % remove sigma_X2 if param.S_tn_w_AM = 0 else it's 1 and will use sigma_X2
        result.S_tn=sigma_X2^(param.S_tn_w_AM)*10^(-SNR_TX/10)*(abs(fft(hext))).^2/param.fb; % this corresponds to +/- pi
        result.S_tn_rms = sqrt(sum(result.S_tn)* delta_f);
    else
        result.S_tn=result.S_tn.*H_rxffe_2;
        result.S_tn_rms = sqrt(sum(result.S_tn)* delta_f);
    end
    %% S_jn from eq 178A-17  Srj_jn from eq 178A-31
    %%  RxFFE,CTLE, and TxFFE was applied in Apply_EQ when called after optimize_fom
    %% Power spectral density of noise due to jitter
    %% Eq. 93A-28 %%
    if ~OP.COMPUTE_COM
        sampling_offset = mod(cursor_i, M);
        %ensure we can take early sample
        if sampling_offset<=1
            sampling_offset=sampling_offset+M;
        end
        if (OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN)
            cursors_early_sample = h(cursor_i-1+M*(-1:param.ndfe));
            cursors_late_sample = h(cursor_i+1+M*(-1:param.ndfe));
        else
            cursors_early_sample = h(sampling_offset-1:M:end);
            cursors_late_sample = h(sampling_offset+1:M:end);
        end
        % ensure lengths are equal
        cursors_early_sample = cursors_early_sample(1:length(cursors_late_sample));
        h_J = (cursors_late_sample-cursors_early_sample)/2*M;
        h_J=reshape(h_J,1,[]); % make row vectors
        if num_ui>length(h_J)
            h_J=[h_J zeros(1,num_ui-length(h_J))];
        else
            h_J=h_J(1:num_ui);
        end
        result.S_jn=sigma_X2*(param.A_DD^2+param.sigma_RJ^2)*(abs(fft(h_J))).^2/param.fb; % this corresponds to +/- pi
        result.S_jn_rms = sqrt(sum(result.S_jn)* delta_f);
        result.S_rj_jn=sigma_X2*(param.sigma_RJ^2)*(abs(fft(h_J))).^2/param.fb; % this corresponds to +/- pi
        result.S_rj_rms = sqrt(sum(result.S_rj_jn)* delta_f);
    else
        result.S_jn=result.S_jn.*H_rxffe_2;
        result.S_jn_rms = sqrt(sum(result.S_jn)* delta_f);
        result.S_rj_jn= result.S_rj_jn.*H_rxffe_2;
        result.S_rj_rms = sqrt(sum(result.S_rj_jn)* delta_f);
    end

    %%  Quantization Noise (S_qn)
    if(param.N_qb ~=0) % "if" to "else" section changed by Hossein Shakiba to implement commit request 4p9_1
        % Take pulse with CTLE and TxFFE and sample it
        pulse_for_quantization = chdata(1).pulse_response_w_CFT_TXFFE_noRxFFE;
        sample_idx = mod(cursor_i-1, M)+1;
        sampled_pulse_response = pulse_for_quantization(sample_idx:end);
        sampled_pulse_response = reshape(sampled_pulse_response,1,[]);
        sampled_pulse_response = sampled_pulse_response(1:M:end);
        if num_ui > length(sampled_pulse_response)
            sampled_pulse_response(end+1:num_ui) = 0;
        else
            sampled_pulse_response = sampled_pulse_response(1:num_ui);
        end

        if strcmpi(param.clip_method, 'slow')
            sig_aftert_ctle_pdf = get_pdf_from_sampled_signal(sampled_pulse_response,param.levels,OP.BinSize);
            noise_after_ctle_pdf = sig_aftert_ctle_pdf;
            sigma_noise = sqrt(result.S_in_rms.^2+result.S_rn_rms^2+result.S_xn_rms^2+result.S_tn_rms^2+result.S_rj_rms^2);
            noise_after_ctle_pdf.y = 1/(sqrt(2*pi)*sigma_noise)*exp(-noise_after_ctle_pdf.x.^2/(2*sigma_noise^2))*OP.BinSize;
            sig_noise_after_ctle_pdf= conv_fct(sig_aftert_ctle_pdf,noise_after_ctle_pdf);
            sig_noise_after_ctle_cdf = cumsum(sig_noise_after_ctle_pdf.y);
            ctle_signal_sigma = sqrt(sum((sig_noise_after_ctle_pdf.x.^2).*sig_noise_after_ctle_pdf.y));
            adc_clip=-CDF_inv_ev(param.P_qc, sig_noise_after_ctle_pdf,sig_noise_after_ctle_cdf);
            result.ctle_signal_sigma=ctle_signal_sigma;
        else
            adc_clip = sum(abs(sampled_pulse_response));
        end
        adc_lsb = 2*adc_clip/(2^param.N_qb-1);
        sigma_Q = adc_lsb/sqrt(12);
        S_qn = sigma_Q^2/f_b*ones(1, num_ui);
        result.adc_clip=adc_clip;
        result.S_qn = S_qn;
        result.S_qn_rms = sqrt(sum(result.S_qn)* delta_f);
    else
        result.S_qn=0;
        result.S_qn_rms = 0;
    end
    result.S_n=result.S_rn+ result.S_tn+ result.S_xn+ result.S_jn+ result.S_qn+result.S_in;
    result.S_n_rms = sqrt(sum(result.S_n)* delta_f);

    %%
    %% Hisi to be included in MLSE rho eq 178a-28
    if OP.COMPUTE_COM
        %% Hisi psd h include CTLE(CFT), TxFFE, and RxFFE but not sigma_X2
        % sampling_offset = mod(cursor_i-1, M)+1; % Commit request 4p4_6, healey_3dj_COM_01_240416
        % hisi=h(sampling_offset:M:end);
        % hisi=hisi(:).';
        % if num_ui>length(hisi)
        %     hisi=[hisi zeros(1,num_ui-length(hisi))];
        % else
        %     hisi=hisi(1:num_ui);% sometime cable channels need a bigger num_ui. prehap compare to the tripple transit time and compute num_ui
        % end
        % cursor_n=find(hisi==max(hisi),1','first');
        samp_idx = (mod(cursor_i-1,M)+1):M:length(h);% Commit request 4p5_2, healey_3dj_COM_01_240521.pdf
        cursor_n=find(samp_idx == cursor_i);
        hisi=h(samp_idx);
        hisi(end+1:num_ui)=0;
        hisi=reshape(hisi(1:num_ui),1,[]);
        %% Eq 178a-29
        for ii=1:length(hisi)
            if ii==cursor_n % cursor
                cursor=hisi(ii);
                hisi(ii)= 0;
            elseif ii >= cursor_n+1 && ii <=cursor_n+Nb
                ib_indx=ii-cursor_n;
                if     hisi(ii) >= bmax(ib_indx)*cursor
                    hisi(ii) = hisi(ii) - bmax(ib_indx)*cursor;
                elseif hisi(ii) <= bmin(ib_indx)*cursor
                    hisi(ii) = hisi(ii) - bmin(ib_indx)*cursor;
                else
                    hisi(ii)=0;
                end
            end
        end
        result.S_isi=sigma_X2*(abs(fft(hisi))).^2/param.fb;
        result.S_isi_rms = sqrt(sum(result.S_isi)* delta_f);
        %%
        result.S_G=result.S_tn+ result.S_rj_jn  + result.S_rn + result.S_in; % eq 178A-30
        result.S_G_rms = sqrt(sum(result.S_G)* delta_f);
        result.Sn_rho=result.S_isi +result.S_n; % need to include xtalk and isi
        result.Sn_rho_rms = sqrt(sum(result.Sn_rho)* delta_f);
    end
end
