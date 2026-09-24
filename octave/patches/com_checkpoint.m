function com_checkpoint(stage, varargin)
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
% ADDED (2026-09-23): the checkpoint harness. Saves whole structs at the
% pipeline boundaries of the main function so SiCoPR can be checked against
% everything the reference holds at each stage, not only the numbers it
% reports.
%
% A no-op unless the environment variable COM_CHECKPOINT_DIR names a
% directory, which the release never sets: the first statement returns. That
% is what keeps this inside the generator's rule that no edit may change a
% number under MATLAB. An environment variable rather than an OP field
% because OP is built inside main from the configuration workbook, so a new
% OP field cannot be set from outside without also patching the config
% reader.
%
%   com_checkpoint('05_optimize_fom_pc1', 'fom_result', fom_result)
%
% writes <COM_CHECKPOINT_DIR>/05_optimize_fom_pc1.mat holding one variable per
% name/value pair. A stage reached twice in one run (the Rx calibration loop
% repeats main's body) gets _2, _3 appended rather than overwriting. A save
% that fails is reported on stderr and does not stop the run: the reference
% result must not depend on whether it was being observed.
d = getenv('COM_CHECKPOINT_DIR');
if isempty(d)
    return
end
S = struct();
for k = 1:2:numel(varargin)
    S.(varargin{k}) = varargin{k+1};
end
base = fullfile(d, stage);
f = [base '.mat'];
n = 1;
while exist(f, 'file')
    n = n + 1;
    f = sprintf('%s_%d.mat', base, n);
end
try
    save('-v7', f, '-struct', 'S');
catch err
    fprintf(2, 'com_checkpoint: %s not saved: %s\n', f, err.message);
end
end
