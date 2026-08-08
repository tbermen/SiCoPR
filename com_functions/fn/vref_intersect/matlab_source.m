function line_intersection=vref_intersect(eye_contour,x_in,vref)
%slope of the 2 sample points around vref crossing
m1=(eye_contour(x_in,1)-eye_contour(x_in-1,1));
%x-intercept for the line
b1=eye_contour(x_in,1)-m1*x_in;
% drawing a horizontal line through vref so slope = 0
m2=0;
%special case for horizontal line, b=y
b2=vref;
%the x-value of line intersection = (b2-b1)/(m1-m2)
%sinc m2 is always 0 and b2 is always vref, this could be stated as (vref-b1)/m1
%And usually vref is 0, so it further reduces to -b1/m1
line_intersection=(b2-b1)/(m1-m2);
