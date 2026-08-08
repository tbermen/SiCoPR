function H_r = OptFom_Calc_Hr(f, param, OP)

%% Combine response of all filters
H_bt=Bessel_Thomson_Filter(param,f,OP.Bessel_Thomson);
H_bw=Butterworth_Filter(param,f,OP.Butterworth);
H_RCos=Raised_Cosine_Filter(param,f,OP.Raised_Cosine);% experiment with RCos
H_r=H_bw.*H_bt.*H_RCos;
