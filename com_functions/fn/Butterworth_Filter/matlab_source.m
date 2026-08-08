function H_bw=Butterworth_Filter(param,f,use_BW)
if use_BW
    H_bw = 1./polyval([1 2.613126 3.414214 2.613126 1], 1i*f./(param.fb_BW_cutoff*param.fb));
else
    H_bw=ones(1,length(f));
end
