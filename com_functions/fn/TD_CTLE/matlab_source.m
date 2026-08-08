function [impulse_response, p1_ctle, p2_ctle, z_ctle] = TD_CTLE(ir_in, fb, f_z, f_p1, f_p2, kacdc_dB, oversampling)
%% Equation 93A-22 implemented in z-space and applied to the impulse response.
p1_ctle = -2*pi*f_p1;
p2_ctle = -2*pi*f_p2;
z_ctle = -2*pi*f_z*10^(kacdc_dB/20);
k_ctle = -p2_ctle;
bilinear_fs = 2*fb*oversampling;
p2d = (1+p2_ctle/bilinear_fs)./(1-p2_ctle/bilinear_fs);
p1d = (1+p1_ctle/bilinear_fs)./(1-p1_ctle/bilinear_fs);
zd = (1+z_ctle/bilinear_fs)./(1-z_ctle/bilinear_fs);
% kd = (bilinear_fs-z_ctle)/((bilinear_fs-p1_ctle)*(bilinear_fs-p2_ctle));
% allow for different pole zeros RIM 9-29-2015
kd = (bilinear_fs-z_ctle)./((bilinear_fs-p1_ctle)*(bilinear_fs-p2_ctle))*f_p1/f_z;
B_filt =k_ctle*kd*poly([zd, -1]);
A_filt=poly([p1d, p2d]);
impulse_response=filter(B_filt,A_filt,ir_in);
