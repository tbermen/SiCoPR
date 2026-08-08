function cdf=pdf_to_cdf(pdf)
%Transform PDF to CDF
%The CDF is natively calculated from negative-to-positive voltage.
%This only gives BER calculation for bottom eye.  Need to also
%calculate a CDF of reversed PDF to get top eye.  The final CDF is the
%min of top and bottom CDF values.
%If only interested in one side, a simple cumsum on y is all that is needed.

cdf.yB=cumsum(pdf.y);
cdf.yT=fliplr(cumsum(fliplr(pdf.y)));
cdf.y=min([cdf.yB(:) cdf.yT(:)],[],2);
cdf.x=pdf.x;
