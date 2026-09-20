function on = com_octave_accel_on()
%% License Notice
%
% Copyright 2026 Todd Bermensolo
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
%
% ADDED (2026-09-18): whether the optional compiled kernels are to be used.
% True only under Octave, when com_octave_accel.oct is on the path, loads, and
% reports the version this file was written against, and COM_OCTAVE_ACCEL is not
% '0'. The kernels return exactly what the interpreted code returns
% (octave/accel/), so this changes run time, never a result; COM_OCTAVE_ACCEL=0
% is how that is checked. Decided once per session.
%
% A .oct built for another Octave version or another platform fails to load, and
% that is an error, not a false: hence the try. Anything the kernel is not sure
% about leaves the release running its own interpreted code, which is the whole
% point of keeping it optional.
persistent cached
if isempty(cached)
    cached = false;
    if exist('OCTAVE_VERSION', 'builtin') && exist('com_octave_accel', 'file') == 3 ...
            && ~strcmp(getenv('COM_OCTAVE_ACCEL'), '0')
        try
            cached = strcmp(com_octave_accel('version'), 'com_octave_accel 1 (2026-09-18)');
        catch
            cached = false;
        end
    end
end
on = cached;
end
