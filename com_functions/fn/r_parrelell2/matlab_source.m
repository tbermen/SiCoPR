function S =r_parrelell2(zref,f,rpad)
S.Parameters(1,1,:) =  -zref/(rpad*(zref/rpad + 2)).*ones(1,length(f));
S.Parameters(2,2,:) =  -zref/(rpad*(zref/rpad + 2)).*ones(1,length(f));
S.Parameters(2,1,:) =   2/(zref/rpad + 2).*ones(1,length(f));
S.Parameters(1,2,:) =  2/(zref/rpad + 2).*ones(1,length(f));
