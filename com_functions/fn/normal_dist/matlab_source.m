function pdf = normal_dist(sigma,nsigma,binsize)
pdf.BinSize=binsize;
pdf.Min=-round(2*nsigma*sigma/binsize); % RIM 03/03/2023 capture more of the tails
pdf.x=(pdf.Min:-pdf.Min)*binsize;
pdf.y=exp(-pdf.x.^2/(2*sigma^2+eps));
pdf.y=pdf.y/sum(pdf.y);
