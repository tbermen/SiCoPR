function [s11, s12, s21, s22] = synth_tline(f, Z_c, Z_0, gamma_coeff, tau, d)
f_GHz=f/1e9;
%% Equation 93A-10 %%
gamma_1 = gamma_coeff(2)*(1+1i);
%% Equation 93A-11 %%
gamma_2 = gamma_coeff(3)*(1-2i/pi*log(f_GHz)) + 2i*pi*tau;
%% Equation 93A-9 %%
gamma = gamma_coeff(1)+gamma_1.*sqrt(f_GHz)+gamma_2.*f_GHz;
gamma(f_GHz==0) = gamma_coeff(1);

%% Equation 93A-12 %%
if d==0
    %force matched impedance if length is 0
    %otherwise divide by zero can occur if Z_c=0
    rho_rl=0;
else
    rho_rl=(Z_c-2*Z_0)/(Z_c+2*Z_0);
end

exp_gamma_d = exp(-d*gamma);
%% Equations 93A-13 and 93A-14 %%
s11 = rho_rl*(1-exp_gamma_d.^2)./(1-rho_rl^2*exp_gamma_d.^2);
s21 = (1-rho_rl^2)*exp_gamma_d./(1-rho_rl^2*exp_gamma_d.^2);
s12 = s21;
s22 = s11;
