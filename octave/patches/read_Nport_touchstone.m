function [sch,schFreqAxis,port_order]=read_Nport_touchstone(touchstone_file,port_order,Z_renorm)
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
[file_path,root_name,extension]=fileparts(touchstone_file);

%fetch number of ports from extension
num_ports = str2double(regexp(extension,'\d+','match','once'));

%% ===== ROBUST READER (FIXED) =====
% The whole file is read at once and parsed with sscanf; textscan is not used
% at all.
%
% WHY (2026-09-16): Octave's textscan can STOP PART WAY THROUGH a touchstone
% file, silently, whether it reads an open file handle or the file's text as a
% string. No error, no short-read indication -- the data simply ends. The
% lines where it stops are ordinary: they parse in full when they are the only
% thing in the file. Measured on the 2026-09-15 corpus,
% CR_1mOSFPDAC_TP0TP5_23p5dB_PCBHost_3p7dB_THRU.s4p: 2180 of its 8001 points,
% the same from a handle or a string; the FEXT1 file beside it, 928 of 8001.
%
% Audited file by file, old reader against this one: 102 of that corpus's
% 1650 files are cut short. 66 are cut above 67 GHz, where COM's flim drops
% the data anyway, so they change nothing -- 65 of them stop at 67.1 GHz, a
% near miss rather than a margin. 36, on ten channels from two contributors,
% are cut between 3.3 and 32 GHz. No file of the 208-case corpus is cut.
%
% The damage is worst when it is quiet. A truncated crosstalk file trips the
% caller's "different number of frequency points" check and stops the run; a
% truncated THRU passes every check and yields COM tens of dB wrong -- 9 cases
% came out at -12 to -23.5 dB where the Python port says +3.2 to +5.8 dB.
%
% Reading the file's text whole reads every file in the corpus completely, and
% subsumes the earlier blank-line shim (approved Gate 3c): blank lines are
% only whitespace between tokens, so the NaN-per-blank-line stream that shim
% filtered never arises.
%
% SECOND DEFECT, measured 2026-09-20: Octave's textscan does not only stop
% early, it also PARSES INACCURATELY. On real touchstone text its '%f' lands up
% to one ulp from the nearest double: of twelve values taken from an 802.3dj
% channel, nine differed from the correctly rounded value, while str2double
% matched all twelve. So even a file the old reader read in full came back with
% last-bit-wrong S-parameters. Reading one 10001-point channel both ways,
% 151195 of 160016 complex entries differ, worst 2.5e-16 absolute (7.3e-16
% relative). That is small and it is not nothing: through the FD metrics it
% shows up around 1e-12, and on COM around 1e-14. An earlier version of this
% comment claimed the two readers parsed the same text to the same doubles.
% That was assumed, not measured, and it was wrong.
%
% SPEED (2026-09-18): the tokens are split with regexp and parsed with
% str2double. The first version of this fix parsed with sscanf over a rejoined
% copy of every line, which is correct but slow under Octave: the parse alone
% took 5.8 s on an 8001-point 4-port file, against 1.4 s here. Across all
% 1814 files of both corpora this reader is 1.8x faster as a whole and returns
% the same S-parameters and frequency axis, bit for bit, on every one. A
% crosstalk case reads up to 13 such files.
txt = fileread(touchstone_file);

%Get option line: the first line whose first non-blank character is '#'
[opt_line, opt_end] = regexp(txt, '^[ \t\r\f\v]*#[^\n]*', 'match', 'end', 'once', 'lineanchors');
if isempty(opt_line)
    error('No option line found in %s', touchstone_file);
end
opt_tokens = textscan(strtrim(opt_line), '%s');
optcell = opt_tokens{1};

body = regexprep(txt(opt_end+1:end), '![^\n]*', '');   % comments out
tokens = regexp(body, '\S+', 'match');
raw = str2double(tokens).';
if any(isnan(raw))
    % sscanf and textscan would stop here, quietly; say which token instead.
    bad = find(isnan(raw), 1);
    error('In %s: value %d, "%s", is not a number', touchstone_file, bad, tokens{bad});
end

columns = num_ports*num_ports*2 + 1;     % values per freq point
num_freq = floor(length(raw) / columns);

if num_freq == 0
    error('No valid S-parameter data found');
end

if mod(length(raw), columns) ~= 0
    % The mainline reader trims silently. Say so instead: this is what a
    % short read looks like, and it is what went undiagnosed above.
    warning('COM:read_Nport_touchstone:ShortRead', ...
        'In %s: %d values is not a whole number of %d-value points; %d trailing value(s) dropped', ...
        touchstone_file, length(raw), columns, mod(length(raw), columns));
end

raw = raw(1:num_freq * columns);         % trim
raw_input = reshape(raw, columns, num_freq).';
%% =================================

%get the frequency mult
frequency_mult_text=optcell{2};
if(strcmpi(frequency_mult_text,'hz'))
    frequency_mult=1;
elseif(strcmpi(frequency_mult_text,'khz'))
    frequency_mult=1e3;
elseif(strcmpi(frequency_mult_text,'mhz'))
    frequency_mult=1e6;
elseif(strcmpi(frequency_mult_text,'ghz'))
    frequency_mult=1e9;
else
    error('Unsupported format for frequency multiplier %s',frequency_mult_text);
end

%get the RI/MA/DB format
format=optcell{4};

%get Z0
port_impedance=str2double(optcell(6:end))';

%grab frequency
raw_input(:,1)=raw_input(:,1)*frequency_mult;
Spar.F=raw_input(:,1);
Spar.F=transpose(Spar.F(:));

%transform data to real/imag
if(strcmpi(format,'ri'))
    ri_data_2D=raw_input(:,2:2:end)+raw_input(:,3:2:end)*1i;
elseif(strcmpi(format,'ma'))
    mag_data=raw_input(:,2:2:end);
    rad_data=raw_input(:,3:2:end)*pi/180;
    ri_data_2D=mag_data.*cos(rad_data)+mag_data.*sin(rad_data)*1i;
elseif(strcmpi(format,'db'))
    mag_data=10.^(raw_input(:,2:2:end)/20);
    rad_data=raw_input(:,3:2:end)*pi/180;
    ri_data_2D=mag_data.*cos(rad_data)+mag_data.*sin(rad_data)*1i;
else
    error('Format %s is not supported. Use RI MA or DB',format);
end

%transform to 3D
matrix_format=0;

if(matrix_format==0)
    %full
    for j=1:num_ports
        pre_out.sp(j,1:num_ports,:)=transpose(ri_data_2D( :,(j-1)*num_ports+1:j*num_ports));
    end
elseif(matrix_format==1)
    %upper
    used_ports=0;
    for j=1:num_ports
        stated_ports=num_ports-j+1;
        pre_out.sp(j,j:num_ports,:)=transpose(ri_data_2D(:,used_ports+1:used_ports+stated_ports));
        pre_out.sp(j:num_ports,j,:)=transpose(ri_data_2D(:,used_ports+1:used_ports+stated_ports));
        used_ports=used_ports+stated_ports;
    end
elseif(matrix_format==2)
    %lower
    used_ports=0;
    for j=1:num_ports
        stated_ports=j;
        pre_out.sp(j,1:j,:)=transpose(ri_data_2D(:,used_ports+1:used_ports+stated_ports));
        pre_out.sp(1:j,j,:)=transpose(ri_data_2D(:,used_ports+1:used_ports+stated_ports));
        used_ports=used_ports+stated_ports;
    end
else
    error('Matrix format not supported');
end

%2-port swap
two_port_swap=1;
if(num_ports==2 && two_port_swap==1)
    temp=pre_out.sp(1,2,:);
    pre_out.sp(1,2,:)=pre_out.sp(2,1,:);
    pre_out.sp(2,1,:)=temp;
end

Spar.S=pre_out.sp;
Spar.Z0=transpose(port_impedance(:));

if length(Spar.Z0)>1
    error('Each port must have the same reference impedance');
end

if ~isequal(Spar.Z0,Z_renorm)
    fprintf('INFO: S-parameter reference impedance of %0.6g ohms renormalized to %0.6g ohms\n',Spar.Z0,Z_renorm);
    rho=(Z_renorm-Spar.Z0)/(Z_renorm+Spar.Z0);
    p=num_ports;
    s_old=Spar.S;
    for k=1:num_freq
        Spar.S(:,:,k)=inv(eye(p)-rho*s_old(:,:,k))*(s_old(:,:,k)-rho*eye(p));
    end
end

%COM formatting
sch=shiftdim(Spar.S,2);

if isempty(port_order)
    port_order = auto_port_order(sch, Spar.F);
end

sch=sch(:,port_order,port_order);
schFreqAxis=Spar.F;

end