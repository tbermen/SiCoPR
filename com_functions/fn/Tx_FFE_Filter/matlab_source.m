function [H_TxFFE] = Tx_FFE_Filter(varargin)
% Author: Richard Mellitz
% Date: 7/29/2022
% generate FD tx ffe system function
% varagins...
% param - stucture
%     param.fb baud rate
%     param.Tx_FFE Tx FFE coef
% f - freq array
% Use_Tx_FFE = flag to use or not
% H_TxFFE is system function for Tx_FFE
db = @(x) 20*log10(abs(x));
[param,varargin]=varargin_extractor(varargin{:});
[f,varargin]=varargin_extractor(varargin{:});
[Use_Tx_FFE,varargin]=varargin_extractor(varargin{:});
if isempty(Use_Tx_FFE)
    Use_Tx_FFE=0;
end
if isempty(param)
    param.fb=106.25e9;
    Tx_FFE=[1 ];
else
    if ~isfield(param, 'Pkg_TXFFE_preset')
        Tx_FFE=[  1 ];
    else
        Tx_FFE=param.Pkg_TXFFE_preset;
    end
end
if isempty(f)
    f=0:10e6:param.fb;
end

if Use_Tx_FFE ~=0
    [mcur,icur] = max(Tx_FFE);
    H_TxFFE=zeros(1,length(f));
    for ii=1:length(Tx_FFE)
        H_TxFFE = Tx_FFE(ii).*exp(-1j*2*pi*(ii-icur).*f/param.fb)+H_TxFFE;
    end
else
    H_TxFFE=ones(1,length(f));
end
