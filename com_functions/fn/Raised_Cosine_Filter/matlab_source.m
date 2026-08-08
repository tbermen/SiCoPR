function H_tw=Raised_Cosine_Filter(param,f,use_RC)
if use_RC
    H_tw = Tukey_Window(f,param,param.RC_Start, param.RC_end);
else
    H_tw=ones(1,length(f));
end
