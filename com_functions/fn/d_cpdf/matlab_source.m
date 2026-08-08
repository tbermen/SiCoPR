function pdf=d_cpdf( binsize, values, probs)
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

%speed up for initializing empty pdf
if all(values==0)
    pdf.BinSize=binsize;
    pdf.Min=0;
    pdf.y=1;
    pdf.x=0;
    return;
end

if ~issorted(values)
    [values,si]=sort(values);
    probs=probs(si);
end
values=binsize*round(values/binsize);
t=(values(1):binsize:values(end));
pdf.Min=values(1)/binsize;
pdf.y=zeros(size(t));
for k=1:length(values)
    if k==1
        bin=1;
    elseif k==length(values)
        bin=length(t);
    else
        [UNUSED_OUTPUT, bin]=min(abs(t-values(k))); %#ok<ASGLU>
    end
    pdf.y(bin) = pdf.y(bin)+probs(k);
end

pdf.BinSize=binsize;
pdf.y=pdf.y/sum(pdf.y);
if any(~isreal(pdf.y)) || any(pdf.y<0)
    error('PDF must be real and nonnegative');
end
support=find(pdf.y);
pdf.y=pdf.y(support(1):support(end));
pdf.Min=pdf.Min+(support(1)-1);
pdf.x=(pdf.Min:-pdf.Min)*binsize;
