function varargout = textscan(fid, fmt, varargin)
% textscan shim for running the COM release file under GNU Octave.
%
% WHY THIS EXISTS
%   read_Nport_touchstone (line 9900 onward of the release file) reads a
%   touchstone file with
%       raw = textscan(fid,'%f %f %f %f %f %f %f %f %f', ...
%                      'CollectOutput',true,'CommentStyle','!');
%   and then locates the frequency lines by counting NaN per ROW:
%       a = sum(isnan(raw{1}),2);  b = find(a==0);  num_freq = length(b);
%
%   That works only if a line break ENDS the current format cycle and pads the
%   remaining conversions with NaN, which is MATLAB textscan behaviour. An .s4p
%   frequency block is one 9-number line (freq + 4 complex pairs) followed by
%   three 8-number lines, so under MATLAB exactly the frequency lines fill all
%   nine columns and `a==0` counts them.
%
%   Octave textscan ignores line boundaries and streams every number into a
%   continuous 9-wide matrix. Measured on a real 10001-point .s4p:
%       octave           36671 x 9, 36670 rows with zero NaN
%       MATLAB would be  40004 x 9, 10001 rows with zero NaN
%   so num_freq comes out 36670 instead of 10001 and the next line raises
%       error: reshape: can't reshape 330033x1 array to 33x36670 array
%
%   This is the first failure after the verLessThan shim and it is fatal for
%   every case, since every case reads at least one .s4p.
%
%   Upstream solves it differently: on the Octave_compat branch Rich Mellitz
%   rewrote read_Nport_touchstone to read a flat %f stream and reshape by a
%   computed count, removing the dependency on the line-record semantics. That
%   rewrite cannot be injected here, because read_Nport_touchstone is a LOCAL
%   subfunction of the monolithic release file and local functions take
%   precedence over anything on the path. Shadowing textscan is the only
%   available override point that does not edit his file.
%
% WHAT IT DOES
%   Only the numeric multi-conversion case is handled here, which is the one
%   read_Nport_touchstone uses. Every other call, including the '%s' option-line
%   read on the preceding lines, is delegated unchanged to the Octave builtin.
%   For the handled case it applies MATLAB record semantics: one record per
%   line, remaining conversions padded with NaN.
%
% WHAT IT COULD GET WRONG
%   Unlike the verLessThan shim, this one moves data, so it belongs in the SHIM
%   attribution category and must be validated rather than trusted. The
%   validation used here is direct: the S-parameter matrix and frequency axis
%   this produces are compared element by element against the Python port's
%   independent touchstone reader, which is bit-exact against MATLAB across all
%   208 reference cases.
%
% STATUS: RETIRED with candidate A at Gate 3b; kept as evidence. Validated bit-exact against an independent parse on all 164 s4p files before retirement.
%
% Copyright 2026 Todd Bermensolo
% SPDX-License-Identifier: BSD-3-Clause

  nfmt = numel(strfind(fmt, '%'));
  numeric_only = ~isempty(regexp(fmt, '^\s*(%f\s*)+$', 'once'));

  % read_Nport_touchstone calls the option-line form as
  %     [optstr, opt_pos] = textscan(...)
  % so the shim has to pass through however many outputs are asked for.
  if ~numeric_only || nfmt < 2
    varargout = cell(1, max(1, nargout));
    [varargout{:}] = builtin('textscan', fid, fmt, varargin{:});
    return;
  end

  comment = '';
  for k = 1:numel(varargin) - 1
    if ischar(varargin{k}) && strcmpi(varargin{k}, 'CommentStyle')
      comment = varargin{k + 1};
    end
  end

  rows_out = {};
  while true
    ln = fgetl(fid);
    if ~ischar(ln)
      break;
    end
    if ~isempty(comment)
      p = strfind(ln, comment);
      if ~isempty(p)
        ln = ln(1:p(1) - 1);
      end
    end
    v = sscanf(ln, '%f').';
    if isempty(v)
      continue;                      % blank or comment-only line
    end
    row = NaN(1, nfmt);
    n = min(numel(v), nfmt);
    row(1:n) = v(1:n);
    rows_out{end + 1} = row;         %#ok<AGROW>
    if numel(v) > nfmt
      % A line longer than the format would start a second record in MATLAB.
      % Not reached for touchstone data; error rather than silently differ.
      error('textscan shim: line carries %d values, format takes %d', ...
            numel(v), nfmt);
    end
  end

  varargout{1} = {vertcat(rows_out{:})};
  if nargout > 1
    varargout{2} = ftell(fid);   % MATLAB returns the file position
  end
end
