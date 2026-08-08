function pdf=Init_PDF_Fast( EmptyPDF, values, probs)
%  p=cpdf(type, ...)
%
% CPDF is a probability mass function for discrete distributions or an
% approxmation of a PDF for continuous distributions.
%
% cpdf is internally normalized so that the sum of probabilities is 1
% (regardless of bin size).

% Internal fields:
% Min: *bin number* of minimum value.
% BinSize: size of PDF bins. Bin center is the representative value.
% Vec: vector of probabilities per bin.

pdf=EmptyPDF;

rounded_values_div_binsize=round(values/pdf.BinSize);
%values=pdf.BinSize*rounded_values_div_binsize;

pdf.x=pdf.BinSize*rounded_values_div_binsize(1):pdf.BinSize:pdf.BinSize*rounded_values_div_binsize(end);
pdf.Min=rounded_values_div_binsize(1);

pdf.y=zeros(size(pdf.x));
%The rounded values divided by binsize will reveal the bin number if
%pdf.Min is subtracted from it
bin_placement=rounded_values_div_binsize-pdf.Min+1;
%Can avoid one addition by inserting the first probability
%actually helps when calling this 2 million times
pdf.y(bin_placement(1))=probs(1);
for k=2:length(values)
    pdf.y(bin_placement(k)) = pdf.y(bin_placement(k))+probs(k);
end
