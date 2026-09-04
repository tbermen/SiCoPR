function tf = verLessThan(package, version)
% verLessThan shim for running the COM release file under GNU Octave.
%
% WHY THIS EXISTS
%   com_ieee8023_4p16p0_beta1_octave_compat.m line 92 runs
%       if verLessThan('matlab', '7.4.1')
%           error('Matlab version 7.4 or higher required')
%       end
%   unguarded, on every invocation. Octave ships its own verLessThan, which
%   resolves the first argument against the list of INSTALLED OCTAVE PACKAGES
%   and raises
%       verLessThan: package "matlab" is not installed
%   because there is no package called "matlab". Verified on Octave 11.3.0.
%   The call therefore aborts every run before any COM work starts.
%
%   Upstream solves the same problem differently: on Rich Mellitz's
%   Octave_compat branch, src/com_ieee8023_.m wraps the call in
%       if ~OP.OCTAVE ... end
%   That edit is not present in the release/ tree, and the ground rules for this
%   comparison forbid editing his file, so the equivalent is supplied here as a
%   path override placed ahead of the COM directory.
%
% WHY IT CANNOT CHANGE A RESULT
%   It returns a constant. The only caller uses the value to decide whether to
%   raise a "MATLAB too old" error; false means the check passes and execution
%   continues exactly as it does on a modern MATLAB. No COM quantity is derived
%   from it. This keeps the SHIM attribution category empty by construction.
%
%   The guard being defeated protects against MATLAB older than 7.4.1 (R2007a).
%   Octave 11.3.0 provides everything that guard was written to require.
%
% STATUS: RETIRED with candidate A at Gate 3b; kept as evidence of what the official release file needs under Octave.
%
% Copyright 2026 Todd Bermensolo
% SPDX-License-Identifier: BSD-3-Clause

tf = false;

end
