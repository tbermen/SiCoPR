function pdf_out = scalePDF(pdf,scale_factor)
                pdf_out=pdf;
                pdf_out.Min=floor(pdf.Min*scale_factor);
                pdf_out.x=(pdf_out.Min:-pdf_out.Min)*pdf_out.BinSize;
                pdf_out.y=interp1(pdf.x*scale_factor,pdf.y,pdf_out.x);
                pdf_out.y(1)= pdf_out.y(2); % NAN interp work around
                pdf_out.y(end)= pdf_out.y(end-1); % NAN interp work around
                pdf_out.y=pdf_out.y/sum(pdf_out.y);
