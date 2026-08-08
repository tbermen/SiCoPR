function [noise_bottom,noise_top]=cdf_to_ber_contour(cdf,specBER)

%For the given BER, find the top & bottom voltage level in the CDF

%for the top, just find the first index at the spec BER
nidx=find(cdf.y>specBER, 1, 'first');
noise_bottom = cdf.x(nidx);
%for top, flip the cdf.  need to operate on row vector for fliplr:  cdf.y(:)'
nidx=find(fliplr(cdf.y(:)')>specBER, 1, 'first');
%the true index without flipping
nidx=length(cdf.y)-nidx+1;
noise_top = cdf.x(nidx);
