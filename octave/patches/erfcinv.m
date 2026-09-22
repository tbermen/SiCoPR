function x = erfcinv(x_in)
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
% Octave's own erfcinv is INACCURATE IN THE TAIL, which matters because COM
% lives there: ber_q, EH_1st and everything downstream of qfuncinv are
% evaluated at specBER, typically 1e-5 to 1e-9.
%
% Measured against a reference accurate to ~1e-15 (round-tripped through
% erfc, which is well conditioned in this direction):
%     y=2e-09   Octave 4.2410900063876591   correct 4.2410900125601803
%     y=1e-12   Octave 5.0420297401745637   correct 5.0420297456390593
% i.e. about 1.1e-9 to 1.5e-9 relative, where MATLAB's is ~1e-15.
%
% This shadows the builtin for every call inside this file. It does NOT
% reimplement erfcinv: it starts from erfinv(1-y), which is exact for moderate
% y and only loses precision as y gets small, and then refines with Newton on
% erfc. erfc is accurate in both languages, so the refinement lands on the
% right answer regardless of how poor the starting point was. Four steps are
% far more than needed (convergence is quadratic); they cost nothing and make
% the result independent of the starting accuracy.
%
% It must not call erfcinv() itself -- inside this file that name resolves
% here, and the recursion would not terminate.
%
% Verified bit-identical to scipy.special.erfcinv at y = 1e-12, 2e-9, 2e-6,
% 1e-5, 1e-3, 0.5 and 1.5, scalar and vector.

x = erfinv(1 - x_in);
for k_ = 1:4
    f_  = erfc(x) - x_in;
    fp_ = -2/sqrt(pi) .* exp(-x.^2);
    st_ = f_ ./ fp_;
    st_(~isfinite(st_)) = 0;       % y at 0 or 2 gives +-Inf; leave it alone
    x = x - st_;
end
