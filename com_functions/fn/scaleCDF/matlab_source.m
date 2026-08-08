function [pdf_out, cdf_out, scale_factor] = scaleCDF(pdf,delta_com,DER0,A_s)
% scale CDF at DER0 to delta_com
pdf_out=pdf;
P=cumsum(pdf.y);
ider0=find(P>=DER0,1,'first');
anias=pdf.x(ider0)/A_s; % ani/as
new_db = 20*log10(-1/anias)-delta_com;
new_value = -1*1/10^(new_db/20);
scale_factor=1/10^(-delta_com/20);
pdf_out=scalePDF(pdf,scale_factor);
cdf_out=cumsum(pdf_out.y);
