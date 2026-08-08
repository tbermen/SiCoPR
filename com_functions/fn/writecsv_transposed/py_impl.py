import csv


def _val_to_str(v):
    if hasattr(v, '__dict__') or isinstance(v, dict):
        return '[struct]'
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if v is None:
        return ''
    try:
        if len(v) == 0:
            return ''
    except TypeError:
        pass
    return str(v)


def writecsv_transposed(output_args, filename):
    """Write struct fields as two-column CSV (Field, Value) (MATLAB lines 11384-11407)."""
    d = output_args if isinstance(output_args, dict) else vars(output_args)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Field', 'Value'])
        for field, value in d.items():
            writer.writerow([field, _val_to_str(value)])
