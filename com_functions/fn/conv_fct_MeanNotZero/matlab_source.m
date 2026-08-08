function p=conv_fct_MeanNotZero(p1, p2)
if p1.BinSize ~= p2.BinSize
error('bin size must be equal')
end

p=p1;
%p.BinSize=p1.BinSize;
%p.Min=p1.Min+p2.Min;
p.Min=round(p1.Min+p2.Min);	% modified by Yasuo Hidaka, 9/4/2016
p.y=conv2(p1.y, p2.y);

%This is equivalent to (p.Min:p.Min+length(p.y)-1)*p.BinSize
%But it is faster to pre-multiply BinSize instead of multiplying the entire
%vector by BinSize
pMax=p.Min+length(p.y)-1;
p.x =(p.Min*p.BinSize:p.BinSize:pMax*p.BinSize);
