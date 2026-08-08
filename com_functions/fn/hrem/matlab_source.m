function out=hrem(h,index,N_bf,bmaxg)

out=[ h(1:index-1) h(index:index+N_bf-1)- sign(h(index:index+N_bf-1)).*   (min( bmaxg, abs( h(index:index+N_bf-1) )))  h(index+N_bf:end) ];
