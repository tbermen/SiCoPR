function S =s_for_c2(zref,f,cpad)
% S is 2 port s parameters out
S_Parameters(1,1,:) =  -1i*2*pi.*f*cpad*zref./(2+1i*2*pi.*f*cpad*zref);
S_Parameters(2,2,:) =  -1i*2*pi.*f*cpad*zref./(2+1i*2*pi.*f*cpad*zref);
S_Parameters(2,1,:) = 2./(2+1i*2*pi.*f*cpad*zref);
S_Parameters(1,2,:) = 2./(2+1i*2*pi.*f*cpad*zref);
S=sparameters(S_Parameters,f,zref);
