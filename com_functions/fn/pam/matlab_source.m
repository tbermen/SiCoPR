function [ dataout ] = pam( data )
% mapping data usng Grey Coding
for i=1:2:floor(length(data)/2)*2
    if data(i:i+1)==[ -1 -1 ]
        dataout(ceil(i/2)) = -1;
    elseif data(i:i+1)==[ -1 1 ]
        dataout(ceil(i/2)) = -1/3;
    elseif data(i:i+1)==[ 1 1 ]
        dataout(ceil(i/2)) = 1/3;
    elseif data(i:i+1)==[ 1 -1 ]
        dataout(ceil(i/2)) = 1;
    end
end
