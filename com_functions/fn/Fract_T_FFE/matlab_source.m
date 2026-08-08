function [ V0 ] = Fract_T_FFE( V , skew_step)
% skew_step   sub UI skew assuming param.samples_per_ui
% V      input signal
% V0     output signal
% Richard Mellitz 8/17/2021
V0=0;
if iscolumn(V); V=V.';end
ishift=skew_step;
V0=circshift(V',[ishift,0])'+V;
V0=V0/2;
