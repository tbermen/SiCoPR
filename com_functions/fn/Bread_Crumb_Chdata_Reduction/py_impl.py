import copy


def Bread_Crumb_Chdata_Reduction(chdata, fields_file):
    """Remove or keep selected fields in chdata list (MATLAB lines 1038-1082).

    fields_file: text file with first line '#reduce' or '#include',
    remaining lines are field names.
    Returns modified chdata list.
    """
    with open(fields_file, 'r') as fid:
        lines = fid.read().splitlines()

    # Remove blank lines
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return chdata

    mode = lines[0].strip().lower()
    field_names = [ln.strip() for ln in lines[1:] if ln.strip()]

    if mode == '#reduce':
        remove_fields = field_names
    elif mode == '#include':
        all_fields = set()
        for cd in chdata:
            all_fields |= set(vars(cd).keys())
        remove_fields = list(all_fields - set(field_names))
    else:
        raise ValueError(f'Bad first line. Must be "#reduce" or "#include"')

    for cd in chdata:
        for field in remove_fields:
            if hasattr(cd, field):
                delattr(cd, field)

    return chdata
