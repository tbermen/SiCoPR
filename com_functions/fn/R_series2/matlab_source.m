function S =R_series2(zref,f,R)
r=ones(1,length(f))*R;
S.Parameters(1,1,:) =  r./(r + 2*zref);
S.Parameters(2,2,:) =  r./(r + 2*zref);
S.Parameters(2,1,:) = (2*zref)./(r + 2*zref);
S.Parameters(1,2,:) =  (2*zref)./(r + 2*zref);
