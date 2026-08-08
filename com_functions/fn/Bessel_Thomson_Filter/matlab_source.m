function H_bt=Bessel_Thomson_Filter(param,f,use_BT)
if use_BT
    a = bessel( param.BTorder );
    acoef=fliplr( a );
    H_bt =a(1)./ polyval(acoef, (1i*f./(param.fb_BT_cutoff*param.fb)));
else
    H_bt=ones(1,length(f));
end
