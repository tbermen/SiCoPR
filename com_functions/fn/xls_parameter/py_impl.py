_SENTINEL = object()


def xls_parameter(param_sheet, param_name, eval_if_string=False, default_value=_SENTINEL):
    """Case-insensitive lookup in 2D param_sheet (MATLAB lines 11408-11493).

    param_sheet: list-of-lists.
    Returns p (the value in the cell one column to the right of the match).
    Raises KeyError if not found and no default_value was provided.
    Raises ValueError if multiple matches found.
    The SAVE_KEYWORD_FILE block is dead code (always false) and is omitted.
    """
    name_lower = param_name.lower()
    matches = [
        (r, c)
        for r, row in enumerate(param_sheet)
        for c, cell in enumerate(row)
        if isinstance(cell, str) and cell.lower() == name_lower
    ]

    if len(matches) == 0:
        if default_value is _SENTINEL:
            raise KeyError(
                f'Mandatory parameter "{param_name}" is missing or incorrect'
            )
        # MATLAB assigns `p = default_value` and then falls through to the
        # `ischar(p) && eval_if_string` line below, so a *default* that is a
        # string is evaluated too. COM Octave:
        #     xls_parameter(sheet, 'ZZ', true, '1+1')             -> 2
        #     xls_parameter(sheet, 'ZZ', true, '[92 92 ; 70 70]') -> 2x2 matrix
        #     xls_parameter(sheet, 'ZZ', 0,    '1+1')             -> '1+1'
        # Returning here instead handed back the unevaluated string. That is
        # reachable: line 10403 of the reference reads
        #     param.pkg_Z_c = xls_parameter(parameter, 'package_Z_c', true,
        #                                   '[92 92 ; 70 70; 80 80; 100 100]').'
        # so a config without package_Z_c transposed a string.
        p = default_value
    elif len(matches) > 1:
        raise ValueError(
            f'{len(matches)} occurrences of "{param_name}" found. Please recheck spreadsheet'
        )
    else:
        r, c = matches[0]
        p = param_sheet[r][c + 1]

    # NOTE: this is Python's eval, and MATLAB's reads MATLAB. Anything in the
    # array syntax a COM workbook actually uses — '[92 92 ; 70 70]', '[-0.34:.02:0]'
    # — raises SyntaxError here where the reference returns a matrix. The
    # engine's own config reader does not use this function; it calls an
    # inlined variant that routes strings through a MATLAB-syntax evaluator
    # (see xls_parameter_txffe._matlab_eval). Sharing that evaluator needs a
    # home outside a single fn directory, so it is left for the caller of this
    # pass to place.
    if isinstance(p, str) and eval_if_string:
        p = eval(p)  # noqa: S307 — matches MATLAB eval() for parameter loading
    return p
