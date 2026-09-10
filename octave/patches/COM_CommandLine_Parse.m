function [config_file,num_fext,num_next,Remember_keyword,OP,varargin]=COM_CommandLine_Parse(OP,varargin)
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
keywords={'Legacy' 'TD' 'Config2Mat' 'Octave'};
Remember_keyword='Legacy';
OP.TDMODE=false;
OP.GET_FD=true;
OP.OCTAVE = exist('OCTAVE_VERSION','builtin') ~= 0; % SiCoPR: detect Octave; the 'Octave' keyword below still forces it
OP.CONFIG2MAT_ONLY=false;
config_file='';
num_fext=[];
num_next=[];
if ~isempty(varargin)
    if ~ischar(varargin{1})
        error('First input must be a string');
    end
    keyword_idx=find(strcmpi(keywords,varargin{1}));
    if isempty(keyword_idx)
        % No Keyword, use the default (Legacy)
        my_keyword = Remember_keyword;
    else
        %Keyword Mode
        my_keyword=varargin{1};
        Remember_keyword=my_keyword;
        varargin(1)=[];
    end
    
    % first keyword check: set special OP values
    switch my_keyword
        case 'TD'
            OP.TDMODE=true;
            OP.GET_FD=false;
            OP.OCTAVE=false;
        case {'Octave'}
            OP.OCTAVE=true;
        otherwise
            OP.OCTAVE=false;
    end

    % main keyword check:  pull varargin
    switch my_keyword

        case {'Legacy' 'TD' 'Octave'}
            [config_file,varargin]=varargin_extractor(varargin{:});
            new_argument = varargin_extractor(varargin{:});
            if ischar(new_argument) && isempty(str2num(new_argument))
                % special input:  allow num_fext and num_next to be omitted when they are 0
                % the trigger is checking that the new_argument is a char that doesn't convert to a number
                num_fext = 0;
                num_next = 0;
            else
                % normal input:  num_fext and num_next are given
                [num_fext,varargin]=varargin_extractor(varargin{:});
                [num_next,varargin]=varargin_extractor(varargin{:});
            end
        case 'Config2Mat'
            OP.CONFIG2MAT_ONLY=true;
            [config_file,varargin]=varargin_extractor(varargin{:});
    end
end
