import numpy as np


def _item_to_str(v):
    if hasattr(v, '__dict__') or isinstance(v, dict):
        return 'struct'
    if isinstance(v, str):
        return v
    if v is None:
        return ''
    try:
        if len(v) == 0:
            return ''
    except TypeError:
        pass
    arr = np.asarray(v)
    if arr.size == 1:
        return str(arr.flat[0])
    return f'"{np.array2string(arr, separator=" ")}"'


def Write_CSV(output_args, csv_file):
    """Write struct fields as single-row CSV header + data (MATLAB lines 4742-4767)."""
    d = output_args if isinstance(output_args, dict) else vars(output_args)
    fields = list(d.keys())
    values = [_item_to_str(d[k]) for k in fields]
    with open(csv_file, 'w') as fid:
        fid.write(','.join(fields) + '\n')
        fid.write(','.join(values) + '\n')
