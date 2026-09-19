function idx = FOM_rxffe_floating_taps(param,h,H,Nb,Rnn,dw,d,wmax,wmin,bmin,bmax,sigma_X2,isi_start,isi_end)
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
%
% SPEED (2026-09-18): the release calls MMSE_FOM once per candidate tap set,
% about 133 times per equalizer evaluation, and uses only its FOM. Here that
% search-mode work is written into the loop, and what does not change between
% candidates is computed once: Rnn/sigma_X2, eye(Nb), zeros(1,Nb) and the
% fixed-tap count. Every other operation is MMSE_FOM's, in MMSE_FOM's order,
% and each candidate still forms its own H(:,cols)'*H(:,cols). Measured on
% captured inputs: 1.25 to 1.33x on this function, every candidate FOM
% bit-identical to the release's. The caller, MMSE, still solves the chosen
% tap set with MMSE_FOM itself.
%
% Deliberately NOT done: forming H'*H once over all columns and indexing it.
% That is 1.8x here, but the indexed product differs from the per-candidate
% one in the last bits on 135 of 138 tap sets tried (search FOMs move by up to
% 3e-14 dB), so a result would stop being bit-identical to the release.
%
% Second pass, 2026-09-18: three more invariants hoisted -- the RxFFE_cpx
% lookup, the wmax/wmin truncation (Nw is fixed within a group), and the FOM
% numerator R_LM/(levels-1), which the release computes first in the same
% expression, so the division by sigma_e sees the same operands. 1.08 to
% 1.10x more on this function, every candidate FOM still bit-identical.
hisi=h(isi_start:isi_end);
hisi=hisi(param.RxFFE_cpx+1:param.N_bmax);
bank_size = param.N_bf;
num_groups = param.N_bg;
num_isi=length(hisi);
max_isi=num_isi-bank_size+1;
valid_tap_locations=1:max_isi;
all_idx=[];
Nfix = param.RxFFE_cmx+1+param.RxFFE_cpx;
cmx1 = param.RxFFE_cmx+1;
RnnS = Rnn/sigma_X2;
ib = eye(Nb);
zb = zeros(1,Nb);
cpx = param.RxFFE_cpx;
fom_num = param.R_LM/(param.levels-1);
for j=1:num_groups
    best_FOM=ones(1,length(valid_tap_locations))*-Inf;
    Nw = Nfix+length(all_idx)+bank_size;
    wmx = wmax;
    wmn = wmin;
    if Nw<length(wmx)
        wmx=wmx(1:Nw);
        wmn=wmn(1:Nw);
    end
    for k=1:length(valid_tap_locations)
        this_location=valid_tap_locations(k);
        new_idx = [all_idx this_location:this_location+bank_size-1];
        new_idx=sort(new_idx);
        new_idx = new_idx+cpx;
        % ---- MMSE_FOM(param,H,Nb,Rnn,dw,d,wmax,wmin,bmin,bmax,sigma_X2,new_idx), FOM only
        cols = [1:Nfix new_idx+cmx1];
        Hs = H(:,cols);
        R = Hs'*Hs+RnnS(cols,cols);
        Hb = Hs(d+2:d+Nb+1,:);
        h0 = Hs(d+1,:);
        A = [R  -Hb'; -Hb ib];
        C = [h0  zb];
        Ct = C';
        Z = A\Ct;
        S_inv = C*Z;
        wbl = [Z; 1-S_inv]/S_inv;
        w=wbl(1:Nw);
        b=wbl(Nw+1:length(wbl)-1);
        blim = min(bmax(:), max(bmin(:), b));
        if (Nb > 0) && any(b ~= blim)
            wl = [R, -h0'; h0, 0]\[h0'+Hb'*blim; 1];
            w = wl(1:Nw);
        end
        wlim = min(wmx(:)*w(1+dw), max(wmn(:)*w(1+dw), w));
        if any(w ~= wlim)
            wlim = wlim/(h0*wlim);
            if Nb > 0
                b = Hb*wlim;
                blim = min(bmax(:), max(bmin(:), b));
            end
        end
        w=wlim;
        b=blim;
        w_tr = w';
        sigma_e=sqrt(sigma_X2*(w_tr*R*w+1+b'*b-2*w_tr*h0'-2*w_tr*Hb'*b));
        best_FOM(k)=20*log10(fom_num/sigma_e);
    end
    [~,best_FOM_idx]=max(best_FOM);
    start_tap = valid_tap_locations(best_FOM_idx);
    all_idx=[all_idx start_tap:start_tap+bank_size-1];
    remove_range=best_FOM_idx:best_FOM_idx+bank_size-1;
    remove_range(remove_range>length(valid_tap_locations))=[];
    valid_tap_locations(remove_range)=[];
    bad_tap=start_tap-bank_size+1:start_tap-1;
    bad_tap(bad_tap<1)=[];
    for n=1:length(bad_tap)
        bad_tap_idx=find(valid_tap_locations==bad_tap(n));
        if ~isempty(bad_tap_idx)
            valid_tap_locations(bad_tap_idx)=[];
        end
    end
end
idx = all_idx+param.RxFFE_cpx;
idx = sort(idx);
end
