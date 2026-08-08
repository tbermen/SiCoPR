# ============================================================
# MATLAB→Python translation notes for varargin_extractor
# MATLAB lines: 11326–11334
# ============================================================
# varargin: MATLAB cell array of variable args → Python *args tuple.
# isempty(varargin): True when called with no args → len(args)==0.
# out_var=[]: MATLAB empty array → Python None (or [] — None preferred
#   as it clearly signals "no value" to callers).
# varg_out={}: empty cell → empty list [].
# varg_out(1)=[]: delete first element of cell array → args[1:].
# Output: (first_arg_or_None, remaining_args_list).
# Known discrepancy from prior com.py attempt: none found.
# ============================================================


def varargin_extractor(*args):
    """Extract the first element from a variable-argument list.

    Returns (out_var, varg_out):
      out_var  — first argument, or None if args is empty
      varg_out — remaining arguments as a list (empty list if ≤1 arg)
    """
    if len(args) == 0:                    # MATLAB isempty(varargin)
        return None, []
    return args[0], list(args[1:])


if __name__ == "__main__":
    print(varargin_extractor())            # (None, [])
    print(varargin_extractor(42))          # (42, [])
    print(varargin_extractor(1, 2, 3))     # (1, [2, 3])
