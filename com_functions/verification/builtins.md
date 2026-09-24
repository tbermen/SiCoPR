# MATLAB builtin and operator classification

Row type (b) of the verification contract. Every name the reference calls is
classified here, and `tests/test_builtin_classification.py` fails on anything
absent. This is a **gate, not a list**: it is derived from the reference, so a
new MATLAB version that uses a new builtin fails the suite until someone
classifies it. Nothing here depends on anyone remembering to add a row.

`sigma = std(x)` and `sigma = np.std(x)` read alike and are not alike. That
defect was not found by reading, because reading is what produced it. It is
found by being forced to write down, once, what MATLAB's `std` actually does.

## Verdicts

| verdict | meaning |
|---|---|
| `same` | numpy/scipy semantics match MATLAB for the way the reference uses it |
| `differs` | a real semantic difference, with the rule stated. **Any function whose Python touches one of these closes ONLY by an oracle-backed test.** A reading cannot close it |
| `no-numeric-result` | plotting, dialogs, file IO, formatting: no numeric result reaches an engine output |
| `undefined-upstream` | the reference CALLS this and never defines it. A defect in the reference, for the COM ad hoc, not a port failure |
| `local-variable` | a parser residual: really a local the extractor could not see. Cited with its line so the claim is checkable |

A `differs` row must state its rule. A row that records the alarm and throws
away the information is worse than nothing, so the gate fails on a `differs`
with no rule.

## Defects in the reference itself

Found by this gate on its first run, 2026-09-22. Each is a name the reference
calls that does not exist, so the line throws if it is ever reached. Report to
the COM ad hoc; do not "fix" the reference locally.

| name | verdict | rule |
|---|---|---|
| `springf` | undefined-upstream | Line 5996, `error(springf('config file syntax error'))`. A typo for `sprintf`. The config-file syntax error path throws "undefined function" instead of reporting the syntax error |
| `f_HP_P` | undefined-upstream | Line 1592 uses bare `f_HP_P(...)` where six other sites correctly use `param.f_HP_P(...)`. Throws if that CTLE branch is reached |
| `f_HP_Z` | undefined-upstream | Line 1592, same defect as `f_HP_P` on the same line |
| `WIENER_HOPF_MMSE` | undefined-upstream | Called by `force`; defined nowhere in 4p16p0 |
| `conv_fct_TEST` | undefined-upstream | Called but never defined |
| `s2_to_s4` | undefined-upstream | Called by `s_for_c4`; defined nowhere, so the reference cannot execute `s_for_c4` at all |

## Semantic differences: the load-bearing rows

A function whose Python touches one of these closes ONLY against the executed
reference. These are where the port's real defects have come from.

| name | verdict | rule |
|---|---|---|
| `std` | differs | MATLAB normalises by **N-1**; `np.std` defaults to **N**. Use `ddof=1`. On a matrix it is **column-wise**, where `np.std` flattens: pass `axis=0`. Complex input: both use `abs` of the deviation, so `ddof=1` is all it needs. This is the defect that prompted the whole verification contract (d5bff6c). Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `round` | differs | MATLAB rounds **half away from zero**; `np.round` rounds **half to even**. 41 sites. Use `_mround` / `_mround_arr`; NaN and Inf pass through both. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `max` | differs | MATLAB **skips NaN**; `np.max` propagates it. MATLAB `max` of an empty array is empty, numpy raises. Use `_mmax` |
| `min` | differs | MATLAB **skips NaN**; `np.min` propagates it. Use `_mmin` |
| `length` | differs | MATLAB `length` is the **longest dimension**, not the first, and is 1 for a scalar. `len()` is neither. Use `_length` |
| `eps` | differs | Bare `eps` is **`eps(1)` = 2.22e-16**, NOT the smallest positive double. Using `np.finfo(float).tiny` (2.2e-308) is wrong by a factor of 1e292; that defect was live in three inlined `make_pkg` copies |
| `sqrt` | differs | MATLAB returns a **complex** result for a negative real; `np.sqrt` returns NaN. Cast to complex where the argument can go negative |
| `norm` | differs | For a **matrix**, MATLAB's default is the 2-norm (largest singular value); `np.linalg.norm`'s default is **Frobenius**: pass `ord=2`. They agree for vectors, real or complex, and on `1`, `Inf` and `'fro'`. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `any` | differs | MATLAB operates **column-wise** on a matrix and returns a row; `np.any` flattens by default. Pass an explicit axis |
| `all` | differs | Column-wise on a matrix, as `any`. Also: MATLAB `if` on an array is true only when **every** element is nonzero |
| `find` | differs | Returns **1-based, column-major linear** indices; `np.flatnonzero` is 0-based row-major. MATLAB also accepts `find(x, 1, 'first')` and, unlike Octave, any unambiguous prefix such as `'fir'` |
| `sort` | differs | MATLAB puts **NaN last** and its sort is stable; numpy's default quicksort is not stable and NaN placement differs. Use `kind='stable'` |
| `interp1` | differs | `'linear'` returns **NaN outside** the data range unless `'extrap'` is given; `np.interp` **clamps** to the end values. Pass `left=np.nan, right=np.nan`, or `interp1d(..., fill_value='extrapolate')` for `'extrap'`. `'pchip'` and `'spline'` **extrapolate in MATLAB without being asked**; Octave returns NA there (octave/patches/H_interp.m), so Octave is not the oracle outside the range. `PchipInterpolator(x, y)` (extrapolate=True) matches MATLAB. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `sum` | differs | MATLAB accumulates SIMD-blocked, numpy is pairwise, Octave is strictly left to right. Last-bit differences only, but Octave is a **third answer**, not a tiebreak. On a matrix it is **column-wise**, where `np.sum` flattens: pass `axis=0`. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). (values recorded, not asserted) |
| `mod` | differs | Agrees with `np.mod` on sign and, for these inputs, to 1e-12 at a non-integer divisor; but MATLAB `mod(x,0)` returns **x** where numpy returns NaN. Every engine site divides by a positive integer M, where the two agree. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `circshift` | differs | Shifts along the **first non-singleton dimension** by default; `np.roll` **flattens** by default. Pass an explicit axis |
| `reshape` | differs | **Column-major**. numpy is row-major: pass `order='F'` |
| `shiftdim` | differs | Column-major dimension semantics, as `reshape` |
| `squeeze` | differs | MATLAB arrays are **minimum 2-D**, so `squeeze` never produces a 1-D result; numpy does |
| `size` | differs | Always reports **at least 2 dimensions**; a numpy 1-D array reports one |
| `transpose` | differs | `transpose(x)` and `.'` are **non-conjugate**; `'` is the **conjugate** transpose. On complex data the two are not interchangeable, and `.T` is the non-conjugate one: `'` is `.conj().T`, and for a 1-D numpy vector `.conj()`. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24), and by the `ctranspose_to_transpose` mutation operator |
| `sign` | differs | Agrees on reals, but for complex MATLAB gives **z/abs(z)** where `np.sign` uses a different convention |
| `erfcinv` | differs | Accurate in MATLAB; **Octave's is inaccurate in the tail**, so the ORACLE must not be trusted here. Shimmed in `octave/patches/erfcinv.m` (56f42e1). Octave 11.3's builtin at y=1e-12 is 6.2e-9 off `scipy.special.erfcinv`, whose `erfc` round trip is exact to 1e-14; from y=1e-5 up the two agree to 1e-12. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `ifft` | differs | `ifft(X, n, 'symmetric')` forces a real result by assuming Hermitian symmetry; plain `np.real(np.fft.ifft(...))` is not equivalent. Use `np.fft.irfft` |
| `inv` | differs | A singular matrix gives **Inf with a warning**, not an exception. Octave's `A\b` diverges from MATLAB here too; see the `mldivide` row |
| `strcmp` | differs | **Case-sensitive**, and returns false for different lengths rather than broadcasting |
| `strcat` | differs | Removes **trailing whitespace** from character arrays |
| `regexp` | differs | MATLAB's regex dialect is not Python's (token syntax, named groups, `once`) |
| `regexprep` | differs | Same dialect difference as `regexp` |
| `strfind` | differs | Accepts a **numeric** pattern array where Octave refuses one |
| `db` | differs | Defaults to the **voltage** convention, 20*log10, not 10*log10 |
| `conv2` | differs | Shape conventions `'full'`/`'same'`/`'valid'` differ from `scipy.signal.convolve2d` defaults |
| `iscolumn` | differs | A numpy 1-D array is neither a row nor a column; the answer depends on an explicit 2-D shape |
| `isrow` | differs | A numpy 1-D array is neither a row nor a column, so this needs an explicit 2-D shape, exactly as `iscolumn` does |
| `isvector` | differs | True for any Nx1 or 1xN **2-D** array; a numpy 1-D array needs an explicit rule |
| `str2num` | differs | Evaluates its argument as MATLAB code, so it accepts expressions `float()` does not |
| `num2str` | differs | Default formatting is not `str()`; it uses about 5 significant digits for non-integers |
| `sprintf` | differs | `%g` and `%d` formatting of non-integers differ from Python's |
| `median` | differs | MATLAB's `median` propagates NaN; matching `np.median` is right, but `np.nanmedian` is not |
| `mean` | differs | Propagates NaN, as `median`. **Column-wise** on a matrix, where `np.mean` flattens: pass `axis=0`. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `var` | differs | As `std`: **N-1** (`ddof=1`) and **column-wise** on a matrix (`axis=0`). Not called by 4p16p0; classified from the parity run. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `rem` | same | `np.fmod`, including `rem(x,0)` = NaN in both. Not called by 4p16p0; classified from the parity run. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `mldivide` | differs | The operator `\`. Square and nonsingular: `np.linalg.solve`. Overdetermined, full rank: `np.linalg.lstsq` (the least-squares answer is unique). **Exactly singular square**: MATLAB warns and returns **Inf**; Octave returns a minimum-norm answer (`[1 2; 2 4] \ [1; 3]` gives [0.28 0.56]); patched in `octave/patches/mldivide_matlab.m`. **Underdetermined**: MATLAB returns a **basic** solution with at most rank(A) nonzeros (QR with column pivoting); Octave and `np.linalg.lstsq` return the **minimum-norm** one, a different vector. Octave is therefore not the oracle for a wide system either; no engine site solves one (force's non-square branch is dead). Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |

## Same semantics

Matched for the way the reference uses them. A row here is a claim that the
port may use the obvious numpy equivalent directly.

| name | verdict | rule |
|---|---|---|
| `abs` | same | |
| `angle` | same | |
| `atan` | same | |
| `ceil` | same | |
| `complex` | same | |
| `conj` | same | |
| `cos` | same | |
| `cumprod` | same | |
| `cumsum` | same | |
| `diag` | same | |
| `diff` | same | |
| `double` | same | |
| `erfc` | same | |
| `exp` | same | |
| `eye` | same | |
| `factorial` | same | |
| `fft` | same | For vectors, the reference's only use (9 sites). On a **matrix** MATLAB transforms **columns** and `np.fft.fft` the last axis: pass `axis=0`. Values differ from Octave by accumulation order only. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `filter` | same | `scipy.signal.lfilter` matches for the 1-D use here (35 sites, all vectors). On a **matrix** MATLAB filters **columns** and `lfilter` the last axis: pass `axis=0`. Checked live against COM Octave 11.3 in `tests/test_matlab_semantics.py` (2026-09-24). |
| `fliplr` | same | |
| `flipud` | same | |
| `floor` | same | |
| `isempty` | same | |
| `isequal` | same | |
| `isinf` | same | |
| `isnan` | same | |
| `kron` | same | |
| `log` | same | |
| `log10` | same | |
| `NaN` | same | |
| `ones` | same | |
| `poly` | same | |
| `polyfit` | same | |
| `polyval` | same | |
| `prod` | same | |
| `real` | same | |
| `repmat` | same | `np.tile`, noting MATLAB tolerates a NaN count where Python raises |
| `setdiff` | same | both return sorted unique values |
| `sin` | same | |
| `sinc` | same | both use the normalised sin(pi x)/(pi x) |
| `issorted` | same | |
| `toeplitz` | same | |
| `tril` | same | |
| `triu` | same | |
| `unwrap` | same | |
| `xor` | same | |
| `zeros` | same | |
| `numel` | same | total element count, unlike `length` |
| `isnumeric` | same | |
| `ischar` | same | |
| `iscell` | same | |
| `isstruct` | same | |
| `islogical` | same | |
| `isreal` | same | |
| `isstring` | same | |
| `isfield` | same | |
| `fieldnames` | same | |
| `rmfield` | same | |
| `struct2cell` | same | |
| `num2cell` | same | |
| `cellfun` | same | with `UniformOutput` handled explicitly |
| `deal` | same | |
| `char` | same | |
| `cell` | same | |
| `string` | same | |
| `lower` | same | |
| `upper` | same | |
| `strcmpi` | same | the case-insensitive counterpart of `strcmp` |
| `strrep` | same | |
| `strsplit` | same | |
| `strjoin` | same | |
| `str2double` | same | returns NaN on failure, like a guarded `float()` |
| `mat2str` | same | |
| `sscanf` | same | |

## No numeric result

Plotting, dialogs, file IO and environment. Nothing here reaches an engine
output, so a difference cannot change a COM value.

| name | verdict | rule |
|---|---|---|
| `axes` | no-numeric-result | |
| `bar` | no-numeric-result | |
| `close` | no-numeric-result | |
| `csvwrite` | no-numeric-result | |
| `disp` | no-numeric-result | |
| `display` | no-numeric-result | |
| `error` | no-numeric-result | control flow, not a value |
| `eval` | no-numeric-result | a porting hazard, but no numeric semantics of its own |
| `exist` | no-numeric-result | |
| `fclose` | no-numeric-result | |
| `fgetl` | no-numeric-result | |
| `figure` | no-numeric-result | |
| `fileparts` | no-numeric-result | |
| `findobj` | no-numeric-result | |
| `fopen` | no-numeric-result | |
| `fprintf` | no-numeric-result | |
| `fscanf` | no-numeric-result | |
| `fullfile` | no-numeric-result | |
| `get` | no-numeric-result | |
| `isfile` | no-numeric-result | |
| `legend` | no-numeric-result | |
| `linkaxes` | no-numeric-result | |
| `load` | no-numeric-result | |
| `movegui` | no-numeric-result | |
| `msgbox` | no-numeric-result | |
| `onCleanup` | no-numeric-result | |
| `plot` | no-numeric-result | |
| `save` | no-numeric-result | |
| `saveas` | no-numeric-result | |
| `semilogx` | no-numeric-result | |
| `semilogy` | no-numeric-result | |
| `set` | no-numeric-result | |
| `snp2smp` | no-numeric-result | RF Toolbox port reordering; the port does this itself |
| `sparameters` | no-numeric-result | RF Toolbox reader; the port has its own touchstone reader |
| `stem` | no-numeric-result | |
| `subplot` | no-numeric-result | |
| `table` | no-numeric-result | |
| `textscan` | no-numeric-result | |
| `timeseries` | no-numeric-result | |
| `title` | no-numeric-result | |
| `uigetfile` | no-numeric-result | |
| `uitab` | no-numeric-result | |
| `uitabgroup` | no-numeric-result | |
| `verLessThan` | no-numeric-result | environment probe |
| `waitbar` | no-numeric-result | |
| `warndlg` | no-numeric-result | |
| `warning` | no-numeric-result | |
| `writecell` | no-numeric-result | |
| `writetable` | no-numeric-result | |
| `xlabel` | no-numeric-result | |
| `xlim` | no-numeric-result | |
| `xlsread` | no-numeric-result | config workbook reader; the port uses openpyxl |
| `ylabel` | no-numeric-result | |
| `ylim` | no-numeric-result | |

## Parser residuals

Names the extractor could not prove are locals. Each is cited so the claim can
be checked. Keep this section small: a name here is one the gate cannot reason
about, so a growing list means the extractor needs work, not that the list needs
another row.

| name | verdict | rule |
|---|---|---|
| `a` | local-variable | Line 1037, `H_bt = a(1)./polyval(...)`. A local holding filter coefficients; the only other occurrences are in comments |
