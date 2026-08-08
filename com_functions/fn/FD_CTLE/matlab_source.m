function [hctf] = FD_CTLE(freq, f_z, f_p1, f_p2, kacdc_dB)
hctf = ( 10.^(kacdc_dB/20) + 1i*freq/f_z ) ./ ( (1+1i*freq/f_p1) .* (1+1i*freq/f_p2)) ;
