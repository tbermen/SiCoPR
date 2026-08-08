# ============================================================
# MATLAB→Python translation notes for COM_CommandLine_Parse
# MATLAB lines: 1149–1200
# ============================================================
# varargin in MATLAB → *args tuple in Python.
# keywords = {'Legacy','TD','Config2Mat'} (case-insensitive match).
# varargin_extractor inlined as _pop: pops first element from a list.
# str2num(s) == '' check: detect non-numeric string → isempty(str2num) ≈ not try_parse_float.
# MATLAB varargin{1} → Python args[0].
# After keyword consumed, args is passed by reference in MATLAB → use list for mutation.
# Returns: config_file, num_fext, num_next, Remember_keyword, OP, remaining_varargin.
# ============================================================

from types import SimpleNamespace


_KEYWORDS = ('legacy', 'td', 'config2mat')


def _try_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def COM_CommandLine_Parse(OP, *varargin):
    """Parse COM command-line arguments (MATLAB lines 1149-1200).

    Returns (config_file, num_fext, num_next, Remember_keyword, OP, remaining_varargin).
    """
    Remember_keyword = 'Legacy'
    OP = SimpleNamespace(**vars(OP)) if not isinstance(OP, dict) else SimpleNamespace(**OP)
    OP.TDMODE = False
    OP.GET_FD = True
    OP.CONFIG2MAT_ONLY = False
    config_file = ''
    num_fext = None
    num_next = None

    args = list(varargin)

    if args:
        if not isinstance(args[0], str):
            raise TypeError('First input must be a string')
        kw_lower = args[0].lower()
        if kw_lower in _KEYWORDS:
            my_keyword = args[0]
            Remember_keyword = my_keyword
            args.pop(0)
        else:
            my_keyword = Remember_keyword

        # first pass: set special OP values
        if my_keyword.upper() == 'TD':
            OP.TDMODE = True
            OP.GET_FD = False

        # main keyword check: pull args
        kw = my_keyword.lower()
        if kw in ('legacy', 'td'):
            # pop config_file
            config_file = args.pop(0) if args else ''
            # peek at next arg: if it's a non-numeric string, num_fext/num_next default to 0
            if args and isinstance(args[0], str) and not _try_float(args[0]):
                num_fext = 0
                num_next = 0
            else:
                num_fext = args.pop(0) if args else None
                num_next = args.pop(0) if args else None
        elif kw == 'config2mat':
            OP.CONFIG2MAT_ONLY = True
            config_file = args.pop(0) if args else ''

    return config_file, num_fext, num_next, Remember_keyword, OP, args
