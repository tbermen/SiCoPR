function [CDF_inv_ev] = CDF_inv_ev(val,PDF,CDF)
index=find(CDF >= val,1,'first');
if isempty(index)
    CDF_inv_ev=PDF.x(end);
else
    CDF_inv_ev=PDF.x(index);
end
