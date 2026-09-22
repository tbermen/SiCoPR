function H_new=H_interp(S21_old, f_old, f_new, f_b)
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
% MATLAB's interp1 EXTRAPOLATES for the pchip/spline/makima methods: outside
% [f_old(1), f_old(end)] it evaluates the polynomial rather than returning
% NaN. Octave's interp1 returns NA there unless 'extrap' is given, so the
% unpatched reference line silently produced NA magnitudes and phases on any
% f_new point outside the measured band, which then propagate through
% 10.^(mag/20).*exp(1j*ph) into H_new.
%
% The reference's own comment below ("the extrapolation performed here can
% cause the magnitude to explode and go to Inf") is explicit that MATLAB
% extrapolates here, and the H_new(isinf(H_new))=0 line exists to contain it.
% Adding 'extrap' restores the MATLAB behaviour; it does not change any value
% inside the interpolation range.
%
% Octave 11.3: interp1([1 2 3 4],[10 20 35 55],[0.5 2.5 5],'pchip')
%   -> [NA 26.857142857142858 NA]
% with 'extrap'
%   -> [7.0624999999999991 26.857142857142858 79.285714285714278]

% --- Magnitude in dB, smoothed with pchip (or 'spline' if you prefer) ----
mag_db_old  = 20*log10(abs(S21_old));
mag_db_new  = interp1(f_old, mag_db_old, f_new, 'pchip', 'extrap');

% --- Phase: unwrap, then linear interpolation (keeps it smooth) ----------
ph_old  = unwrap(angle(S21_old));
ph_new  = interp1(f_old, ph_old, f_new, 'pchip', 'extrap');

% --- Reassemble complex S21 on the fine grid ----------------------------
H_new = 10.^(mag_db_new/20) .* exp(1j*ph_new);
H_new(isinf(H_new))=0;

% Only need data up to Fb/2.  Zero out anything above Fb/2
% In some cases, the extrapolation performed here can cause the magnitude to explode and go to Inf
% Forcing to zero avoids that instability
inq=find(f_new<=f_b/2,1,'last');
H_new(inq+1:end) = 0;
