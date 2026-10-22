from com_functions.fn.xls_parameter.py_impl import xls_eval as _xls_eval


def xls_parameter_txffe(param_sheet, param_name):
    """Case-insensitive search in 2D param_sheet; return (value, found) (MATLAB lines 11494-11515).

    param_sheet is a list-of-lists (rows × cols).
    Returns (p, 1) on match, (0, 0) if not found.
    Raises ValueError if multiple matches found.
    """
    name_lower = param_name.lower()
    matches = [
        (r, c)
        for r, row in enumerate(param_sheet)
        for c, cell in enumerate(row)
        if isinstance(cell, str) and cell.lower() == name_lower
    ]
    if len(matches) == 0:
        return 0, 0
    if len(matches) > 1:
        raise ValueError(
            f'{len(matches)} occurrences of "{param_name}" found. Please recheck spreadsheet'
        )
    r, c = matches[0]
    p = param_sheet[r][c + 1]
    # ML 12536: p=eval(p). xls_eval reads MATLAB, as the reference does; the
    # Python eval this replaces read '[1 0 0]' as a SyntaxError and ran any
    # Python text (findings F12, F19). COM Octave pins are in test_verify.py.
    if isinstance(p, str):
        p = _xls_eval(p)
    return p, 1
