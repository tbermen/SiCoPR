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
        return default_value

    if len(matches) > 1:
        raise ValueError(
            f'{len(matches)} occurrences of "{param_name}" found. Please recheck spreadsheet'
        )

    r, c = matches[0]
    p = param_sheet[r][c + 1]
    if isinstance(p, str) and eval_if_string:
        p = eval(p)  # noqa: S307 — matches MATLAB eval() for parameter loading
    return p
