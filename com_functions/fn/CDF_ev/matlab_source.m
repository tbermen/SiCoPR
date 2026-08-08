function [CDF_ev] = CDF_ev(val,PDF,CDF)
index=find(PDF.x >= -val,1,'first');
CDF_ev=CDF(index);
