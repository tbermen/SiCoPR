# ============================================================
# MATLAB→Python translation notes for COM_CommandLine_Parse
# MATLAB lines: 1149–1200
# ============================================================
# varargin in MATLAB → *args tuple in Python.
# keywords = {'Legacy','TD','Config2Mat'}: the *match* is strcmpi, so it is
# case-insensitive, but my_keyword keeps the caller's spelling and the two
# switch statements that follow are case-SENSITIVE. So 'td' is consumed as a
# keyword yet selects no case: nothing further is parsed. See the oracle
# evidence beside the switch below.
# varargin_extractor inlined as _pop: pops first element, or MATLAB [] → None.
# str2num(s) is eval(['[' s ']']); isempty(str2num(s)) is approximated by
# _str2num_nonempty, see its comment.
# After keyword consumed, args is passed by reference in MATLAB → use list for
# mutation.
# Returns: config_file, num_fext, num_next, Remember_keyword, OP, remaining_varargin.
# ============================================================

import re
from types import SimpleNamespace


_KEYWORDS = ('Legacy', 'TD', 'Config2Mat')

# Bare words MATLAB's str2num resolves to a numeric (or logical) value, so
# str2num returns non-empty for them.
_NUMERIC_WORDS = frozenset(('pi', 'e', 'i', 'j', 'eps', 'inf', 'nan',
                            'true', 'false'))


def _try_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _str2num_nonempty(s):
    """~isempty(str2num(s)), for the numeric-array forms str2num accepts.

    str2num is eval(['[' s ']']), so anything that evaluates to a numeric
    array is non-empty: a bare number, a numeric constant such as pi or eps,
    a logical, a bracketed or separated list, a colon range. Arithmetic
    expressions ('2+3') are not covered here; they would need an expression
    evaluator and are not a command line COM is given.
    """
    t = s.strip()
    if t.startswith('[') and t.endswith(']'):
        t = t[1:-1]
    tokens = [k for k in re.split(r'[\s,;]+', t) if k]
    if not tokens:
        return False                      # str2num('') and str2num('[]') are []
    for k in tokens:
        parts = k.split(':') if ':' in k else [k]
        for p in parts:
            if not (_try_float(p) or p.lower().lstrip('+-') in _NUMERIC_WORDS):
                return False
    return True


def _pop(args):
    """varargin_extractor: the first argument, or MATLAB [] when there is none."""
    return args.pop(0) if args else None


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
        if args[0].lower() in [k.lower() for k in _KEYWORDS]:
            # Keyword Mode: strcmpi matched, so the keyword is consumed
            # whatever its case, and Remember_keyword keeps that spelling.
            my_keyword = args[0]
            Remember_keyword = my_keyword
            args.pop(0)
        else:
            my_keyword = Remember_keyword

        # first keyword check: set special OP values.
        # COM Octave: switch my_keyword is case-sensitive, so ('td','cfg.csv',0,0)
        # leaves TDMODE=0, GET_FD=1, config_file=[], num_fext=[], num_next=[],
        # and ('config2mat','cfg.xlsx') leaves CONFIG2MAT_ONLY=0 and
        # config_file=[]. Only the exact spellings below do anything.
        if my_keyword == 'TD':
            OP.TDMODE = True
            OP.GET_FD = False

        # main keyword check: pull varargin
        if my_keyword in ('Legacy', 'TD'):
            config_file = _pop(args)
            new_argument = args[0] if args else None
            if isinstance(new_argument, str) and not _str2num_nonempty(new_argument):
                # special input: allow num_fext and num_next to be omitted
                # when they are 0
                num_fext = 0
                num_next = 0
            else:
                # normal input: num_fext and num_next are given
                num_fext = _pop(args)
                num_next = _pop(args)
        elif my_keyword == 'Config2Mat':
            OP.CONFIG2MAT_ONLY = True
            config_file = _pop(args)

    return config_file, num_fext, num_next, Remember_keyword, OP, args
