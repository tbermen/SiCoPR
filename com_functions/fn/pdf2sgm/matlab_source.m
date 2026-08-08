function sgm = pdf2sgm(pdf)
avg = sum(pdf.x .* pdf.y);
sgm = sqrt(sum((pdf.x - avg).^2 .* pdf.y));
