function [out_var,varg_out]=varargin_extractor(varargin)
if isempty(varargin)
    out_var=[];
    varg_out={};
else
    out_var=varargin{1};
    varg_out=varargin;
    varg_out(1)=[];
end
