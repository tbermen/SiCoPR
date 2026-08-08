function out_pdf=combine_pdf_same_voltage_axis(pdf1,pdf2)
if pdf1.BinSize ~= pdf2.BinSize
    error('bin size must be equal')
end

x1=pdf1.x;
y1=pdf1.y;
x2=pdf2.x;
y2=pdf2.y;
%find the pdf with a larger min, force it to have the same min, and insert
%probability = 0
min1=pdf1.x(1);
min2=pdf2.x(1);
shift_amount=round(abs(min1-min2)/pdf1.BinSize);
if min1<min2
    x2=[pdf1.x(1:shift_amount) pdf2.x];
    y2=[zeros(1,shift_amount) pdf2.y];
else
    x1=[pdf2.x(1:shift_amount) pdf1.x];
    y1=[zeros(1,shift_amount) pdf1.y];
end
%find the pdf with smaller max, force it to have the same max, and insert
%probability=0
L1=length(x1);
L2=length(x2);
Ldiff=abs(L1-L2);
if L1>L2
    out_x=x1;
    y2=[y2 zeros(1,Ldiff)];
else
    out_x=x2;
    y1=[y1 zeros(1,Ldiff)];
end
%now the 2 pdfs have the same voltage axis, add probabilities together
%renormalization is not handled here, so the output pdf will not have sum=1
%It is the responsibility of the calling function to handle renormalization
%if needed
out_y=y1+y2;
out_pdf.x=out_x;
out_pdf.y=out_y;
out_pdf.BinSize=pdf1.BinSize;
