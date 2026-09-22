function x = mldivide_matlab(A, b)
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
%% for Octave compatability
%%
%% MATLAB and Octave disagree on a SQUARE system whose matrix is exactly
%% singular. MATLAB's backslash warns "Matrix is singular to working
%% precision" and returns Inf; Octave's returns a minimum-norm least-squares
%% solution instead, which is a different answer, not a rounding difference:
%%     [1 2; 2 4] \ [1; 3]   MATLAB [Inf; Inf]   Octave [0.28; 0.56]
%%
%% Octave's own inv() already carries a singular matrix to Inf exactly as
%% MATLAB does, so the MATLAB behaviour is available here; it just is not what
%% backslash does. rcond is exactly 0 for the singular case and finite
%% otherwise, which is the discriminator.
%%
%% Only the exactly-singular case is redirected. A merely ill-conditioned
%% matrix (rcond 2.8e-12 measured) is left to backslash, because MATLAB also
%% returns a finite answer there, warning but not failing. Verified across
%% exactly-singular/inconsistent, exactly-singular/consistent, near-singular
%% and well-conditioned inputs.
%%
%% This matters for force(): its RxFFE solve is VV'\FV', and until this
%% existed an oracle run of that function on a degenerate VV would have
%% pinned Octave's answer rather than the reference's.

if size(A,1) == size(A,2) && rcond(A) == 0
    w_ = warning('off', 'all');
    x = inv(A) * b;        %% MATLAB: Inf, with the singular warning
    warning(w_);
else
    x = A \ b;
end
