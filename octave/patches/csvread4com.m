function parameters = csvread4com(filename)
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
% OCTAVE compatable CSV reader for COM config files
fid = fopen(filename,'rt');
if fid < 0
    error('Cannot open file: %s', filename);
end
parameters = {};
r = 1;
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    start_idx = 1;
    c = 1;
    for k = 1:length(line)+1
        is_end = (k > length(line));
        if is_end || line(k)==','
            token = strtrim(line(start_idx:k-1));
            if isempty(token)
                parameters{r,c} = NaN;
            else
                num = str2double(token);
                if ~isnan(num)
                    parameters{r,c} = num;
                else
                    parameters{r,c} = token;
                end
            end
            c = c + 1;
            start_idx = k + 1;
        end
    end
    r = r + 1;
end
fclose(fid);
% Pad rows to rectangular shape like xlsread raw output
nrows = size(parameters,1);
maxcols = 0;
for r = 1:nrows
    maxcols = max(maxcols,size(parameters(r,:),2));
end
for r = 1:nrows
    for c = size(parameters(r,:),2)+1:maxcols
        parameters{r,c} = NaN;
    end
end

end