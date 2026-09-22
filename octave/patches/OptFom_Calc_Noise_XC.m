function Noise_XC = OptFom_Calc_Noise_XC(H_low_xc, ctle_gain_xc, SETTINGS, param, OP)
%% License Notice
%
% Copyright 2025 802-COM Authors
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
% - Redistributions of source code must retain the above copyright
%   notice, this list of conditions and the following disclaimer.
% 
% - Redistributions in binary form must reproduce the above copyright
%   notice, this list of conditions and the following disclaimer in the
%   documentation and/or other materials provided with the distribution.
% 
% - Neither the name of the copyright holder nor the names of its
%   contributors may be used to endorse or promote products derived from
%   this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% SPDX-License-Identifier: BSD-3-Clause
% for Octave compatability
%
% Octave's ifft has no 'symmetric' flag and rejects it outright
% ("invalid conversion from string to real scalar"), so the reference line
%     ifft(X, 2*length(X), 'symmetric')
% cannot run here verbatim. It is not unreproducible, only unspelled:
% ifft(X,n,'symmetric') PADS X to n, then uses only the first n/2+1 entries
% and infers the rest by conjugate symmetry. Building that mirrored spectrum
% explicitly and taking real(ifft(...)) is the same operation, and keeps this
% function usable as an oracle.
%
% Note real(ifft([X zeros])) is NOT the same thing: that is the ifft of the
% HERMITIAN PART of the padded vector, and it disagrees by a non-constant
% ratio. Measured against the construction below for N=4:
%     symmetric : 0.281975  0.187496569158  0.065625  0.0250034308417
%     real(ifft): 0.1941125 0.146873284579  0.0859375 0.0656267154208
% The SiCoPR port carried the second form and was corrected to match this.

H_r_xc = SETTINGS.H_r_xc;
f_xc = SETTINGS.f_xc;
N_fft_by2 = SETTINGS.N_fft_by2;
switch upper(OP.FFE_OPT_METHOD)
    case 'WIENER-HOPF'
        % obsolete not used
        H_ctf_xc = H_low_xc.*ctle_gain_xc;
        H_rx_ctle_xc = H_r_xc.*H_ctf_xc;
        Var_eta0 =  param.eta_0*f_xc(end)/1e9;
        X = H_rx_ctle_xc.*conj(H_rx_ctle_xc);
        n = 2*length(H_rx_ctle_xc);
        Xp = [X(:).', zeros(1, n-length(X))];      % ifft pads to n first
        half = Xp(1:floor(n/2)+1);                 % the entries MATLAB keeps
        Xful = [half, conj(half(end-1:-1:2))];     % conjugate-symmetric extension
        XC_rx_ctle = real(ifft(Xful, n));
        Noise_XC = Var_eta0.*XC_rx_ctle(1:param.samples_per_ui:N_fft_by2);

        if OP.Do_White_Noise
            Noise_XC = Noise_XC(1);
        end
    otherwise
        Noise_XC=[];
end
