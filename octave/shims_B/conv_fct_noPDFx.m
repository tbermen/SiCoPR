function p=conv_fct_noPDFx(p1, p2)
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
% Remove return and compuation of PDF.x which causes an excessive run times in Octava
% Only called from MLSE_U1_c_178A
if p1.BinSize ~= p2.BinSize
    error('bin size must be equal')
end

p=p1;
%p.BinSize=p1.BinSize;
%p.Min=p1.Min+p2.Min;
p.Min=round(p1.Min+p2.Min);	% modified by Yasuo Hidaka, 9/4/2016
p.y=conv2(p1.y, p2.y);
%p.x =p.Min*p.BinSize:p.BinSize:-p.Min*p.BinSize;
%p.x =(p.Min:-p.Min)*p.BinSize;	% modified by Yasuo Hidaka, 9/4/2016
pMax=p.Min+length(p.y)-1;
% --- SHIM (approved Gate 3a) -------------------------------------------
% The branch version leaves this line commented out, so p.x keeps the
% PRE-convolution axis inherited from p1 while p.y and p.Min both grow.
% CDF_ev indexes PDF.x, so from the second MLSE loop iteration onward the
% CDF index is drawn from a stale, shorter axis. Measured effect on
% woXtalk_T1_R01: DER_MLSE 9.4e-14 instead of 4.6e-08, delta_COM 4.028
% instead of 1.371, COM 6.153 dB instead of 3.496 dB.
% Restoring the one line that conv_fct already has puts the axis back in
% step with y. Nothing else is changed.
p.x =(p.Min*p.BinSize:p.BinSize:pMax*p.BinSize);
% --- end SHIM ----------------------------------------------------------




