function append_csv_row(file_path, header_cells, row_cells)

file_exists = isfile(file_path);
fid = fopen(file_path, 'a');
if fid == -1
    warning('Could not open %s', file_path);
    return;
end
cleanup = onCleanup(@() fclose(fid));

% Write header once
if ~file_exists
    fprintf(fid, '%s\n', strjoin(header_cells, ','));
end

if ~isempty(row_cells)
    % Convert each cell to CSV-safe text
    out = cell(size(row_cells));
    for k = 1:numel(row_cells)
        v = row_cells{k};
        if isnumeric(v)
            out{k} = sprintf('%.6g', v);
        elseif ischar(v) || isstring(v)
            out{k} = ['"' char(v) '"'];   % quote strings for CSV safety
        else
            out{k} = '""';
        end
    end

    fprintf(fid, '%s\n', strjoin(out, ','));
end
