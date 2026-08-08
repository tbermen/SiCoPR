function csv_string = str2csv(c)
% convert a cell array of strings to a csv string
cell_tmp = cell(2, length(c));
cell_tmp(1,:)=c;
cell_tmp(2,:) = {','};
cell_tmp{2,end} = '';
csv_string=strcat(cell_tmp{:});
