function H_tw=Tukey_Window(f,param,fr,fb)
% fr is the start of the raised cosine window (pass edge)
% fb is the end of the raised cosine window (stop edge)
if ~exist('fr','var') && ~exist('fb','var')
    fb=param.fb;
    fr=param.f_r*param.fb;
end
fperiod=2*(fb-fr);
H_tw = [ ones(1,length(f(f<fr))) ...
    0.5*cos(2*pi*(f( f>=fr & f<=fb) -fb)/fperiod-pi)+.5 ...
    zeros(1,length(f(f>fb))) ];
H_tw=H_tw(1:length(f));
