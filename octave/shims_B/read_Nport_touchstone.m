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
fid=fopen(touchstone_file);

%fetch number of ports from extension
num_ports = str2double(regexp(extension,'\d+','match','once'));

%Get option line
[optstr,~] = textscan(fid,'%s',1,'Delimiter','','CommentStyle','!');
optcell=textscan(optstr{1}{1},'%s');
optcell=optcell{1};
while isempty(optcell) || isempty(strfind(optcell{1},'#'))
    [optstr,~] = textscan(fid,'%s',1,'Delimiter','','CommentStyle','!');
    optcell=textscan(optstr{1}{1},'%s');
    optcell=optcell{1};
end

%% ===== ROBUST READER (FIXED) =====
raw = textscan(fid, '%f', ...
    'CommentStyle','!', ...
    'CollectOutput', true);
fclose(fid);

raw = raw{1};
% --- SHIM (approved Gate 3c) -------------------------------------------
% Octave textscan with a %f format emits one NaN per BLANK LINE. The branch
% reader reshapes the flat stream without filtering NaN, so any touchstone
% file whose frequency blocks are separated by blank lines is misaligned and
% its frequency axis destroyed. 12 of the 164 files in this corpus are like
% that, all belonging to channels R24-R26 (li_dj CR Designs A/B/C).
% Measured on li_dj_CR_Design_C_Rev1_THRU.s4p: 396033 real values + 12000 NaN
% gives floor(408033/33) = 12364 frequency points instead of 12001, with a
% non-monotonic axis. The mainline reader drops NaN explicitly; this restores
% that one line and nothing else. Provably a no-op for files without blanks.
raw = raw(~isnan(raw));
% --- end SHIM ----------------------------------------------------------

columns = num_ports*num_ports*2 + 1;     % values per freq point
num_freq = floor(length(raw) / columns);

if num_freq == 0
    error('No valid S-parameter data found');
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