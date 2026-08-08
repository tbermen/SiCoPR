function [ V0 ] = FFE( C , cmx,spui, V )
% C      FFE taps
% cmx    number of precursors taps
% spui   samples per ui
% V      input signal
%speed ups implemented:
%1) ignore C(i)=0.  No need to circshift and then multiply by 0
%2) change ishift to shift an extra -cmx.  This avoids extra circshift at the end

V0=0;
if iscolumn(V); V=V.';end
for i=1:length(C)
    if C(i)~=0
        ishift=(i-1-cmx)*spui;
        V0=circshift(V',[ishift,0])*C(i)+V0;
    end
end
