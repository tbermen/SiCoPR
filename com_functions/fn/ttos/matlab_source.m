function s_params = ttos(t_params)
% p 67 R. Mavaddat. (1996). Network scattering parameter. Singapore: World Scientific.
[t11, t12, t21, t22] = deal(t_params(1,1,:), t_params(1,2,:), t_params(2,1,:), t_params(2,2,:));
delta = t11.*t22-t21.*t12;
t11(t11==0)=eps;
s_params = [t21./t11, delta./t11; 1./t11, -t12./t11];
