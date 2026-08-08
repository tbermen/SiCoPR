"""Stage 2: Interface contract verification for the critical call chain."""
import ast
import os
import json
import sys

def get_public_signatures(filepath):
    """Return {func_name: (n_positional_args, n_returns)} for public functions."""
    with open(filepath, encoding='utf-8') as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f'  SyntaxError in {filepath}: {e}')
        return {}
    sigs = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith('_'):
            continue
        # Count args (excluding *args and **kwargs); subtract injected _fn kwargs
        args = node.args
        n_pos = len(args.args)
        n_defaults = len(args.defaults)
        # Count non-default args (required positional)
        n_required = n_pos - n_defaults
        # Find first return statement for return count
        n_ret = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                if isinstance(child.value, ast.Tuple):
                    n_ret = len(child.value.elts)
                else:
                    n_ret = 1
                break
        sigs[node.name] = (n_pos, n_required, n_ret)
    return sigs

# Load all py_impl signatures
with open('com_functions/registry.json', encoding='utf-8-sig') as f:
    fns = json.load(f)['functions']

all_sigs = {}
for fn in fns:
    path = os.path.join('com_functions', 'fn', fn['name'], 'py_impl.py')
    if os.path.exists(path):
        sigs = get_public_signatures(path)
        all_sigs.update(sigs)

print(f'Loaded signatures for {len(all_sigs)} public functions.')

# Critical caller→callee pairs: (name, min_required_args, expected_returns)
# min_required_args = number of positional args that MUST be provided (no defaults)
# Use -1 to skip checking that dimension
critical_pairs = [
    # (function_name,   min_required_args,  expected_returns,  notes)
    ('bessel',                          1,  1,  'bessel(n)'),
    ('Bessel_Thomson_Filter',           3,  1,  '(param, f, flag)'),
    ('Butterworth_Filter',              3,  1,  '(param, f, flag)'),
    ('FD_CTLE',                         5,  1,  '(f, fz, fp1, fp2, gdc)'),
    ('read_Nport_touchstone',           3,  2,  '(filename, port_order, ref_Z)'),
    ('parameter_size_adjustment',       2,  1,  '(param, OP)'),
    ('process_sxp',                     4,  2,  '(param, OP, chdata, chdata_xt)'),
    ('COM_FD_to_TD',                    3,  1,  '(chdata, param, OP)'),
    ('optimize_fom',                    5,  1,  '(OP, param, chdata, sigma_bn, do_C2M)'),
    ('Apply_EQ',                        4,  1,  '(BEST, param, OP, chdata)'),
    ('COM_eye_width',                   7,  5,  '(chdata, delta_y, fom_result, param, OP, Struct_Noise, flag)'),
    ('normal_dist',                     3,  1,  '(sigma, nsigma, binsize)'),
    ('pdf_to_cdf',                      1,  1,  '(pdf)'),
    ('get_pdf_from_sampled_signal',     3,  1,  '(x, levels, delta_y)'),
    ('OptFom_Calc_FOM',                 6,  2,  '(chdata, do_C2M, THIS, param, OP, sbr)'),
    ('OptFom_Calc_Noise',               7,  2,  '(THIS, Best_FOM, sbr, SETTINGS, chdata, param, OP)'),
    ('OptFom_Build_TXFFE',              1,  5,  '(param)'),
    ('TDR_ERL_Processing',             -1, -1,  'variable signature — skip return check'),
    ('FD_Processing',                  -1, -1,  'variable signature — skip return check'),
]

failures = []
warnings_list = []

for name, min_req, exp_ret, note in critical_pairs:
    if name not in all_sigs:
        failures.append(f'MISSING: {name} ({note})')
        continue
    n_pos, n_req, n_ret = all_sigs[name]
    if min_req >= 0 and n_req > min_req + 5:  # allow up to 5 injected _fn kwargs
        warnings_list.append(f'  {name}: expected ~{min_req} required args, got {n_req} required (may have extra injections)')
    if exp_ret > 0 and n_ret != exp_ret:
        failures.append(f'  {name}: expected {exp_ret} returns, got {n_ret} ({note})')

if failures:
    print('\nINTERFACE FAILURES:')
    for f in failures:
        print(f'  {f}')
else:
    print('\nAll critical interface checks PASSED.')

if warnings_list:
    print('\nWarnings (non-fatal):')
    for w in warnings_list:
        print(w)

if failures:
    sys.exit(1)
