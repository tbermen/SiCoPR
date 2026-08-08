function [ V0 ] = FFE_Fast( C,V_shift )
% C      FFE taps
% V      input signal separated into length(C) columns with circshift already performed
% This function is only to speed up FFE in optimize_fom.  Since the signal that is being
% shifted is the same for all loops of TXFFE taps, a lot of time can be
% saved by pre-shifting it and remembering it across loops
% Another speed up:  only multiply by indices of C that are not 0

V0=0;
for i=1:length(C)
    if C(i)~=0
        V0=V_shift(:,i)*C(i)+V0;
    end
end
