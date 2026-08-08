function S =s_for_c4(zref,f,cpad)
S2 = s_for_c2(zref,f,cpad);
S4P=s2_to_s4(S2.Parameters);
S=sparameters(S4P,f,zref);
S.Parameters=snp2smp(S.Parameters,zref,[ 1 3 2 4]);
