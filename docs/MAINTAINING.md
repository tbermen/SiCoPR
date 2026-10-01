# Maintaining SiCoPR

The maintainer's side of [`CONTRIBUTING.md`](../CONTRIBUTING.md): what the
commitment is, what happens when a Reference Code release is published, how
issues are handled, and how a state of the repository is released.

---

## 1. The commitment

Stated for users in the README, under
[*What the maintainer commits to*](../README.md#what-the-maintainer-commits-to), and
presented to the IEEE 802.3 COM ad hoc on 2026-09-29. In short:

- The commitment is **consistency and correlation with the Reference Code**, and
  nothing wider. IEEE maintains one COM code base. SiCoPR is an extra
  implementation outside it, not a second standard.
- **The engine changes only for:**
  - a new official Reference Code release;
  - a port defect;
  - a speed-up that `tools/equivalence_check.py` proves changes no result.
- **A change to COM goes to the ad hoc and the Reference Code first.** SiCoPR
  re-correlates after the release.
- **The supporting tools are provided as they are:** `gui/`, `R/`, the study
  tools and the search instrumentation. They never change a COM result and are
  not re-verified with each release.

## 2. When a Reference Code release is published

Only an official release on the IEEE COM Git site starts this. Drafts and
development branches do not.

| step | how | evidence it leaves |
|---|---|---|
| 1. Diff the release against the one emulated now | `python tools/matlab_version_diff.py OLD.m NEW.m`; run `--self-check` first | a `MATLAB_<ver>_CHANGES.md`, as [`MATLAB_4p16p0_CHANGES.md`](MATLAB_4p16p0_CHANGES.md) |
| 2. Map the diff onto the functions to re-check | the differ lists the `py_impl.py` files whose MATLAB changed | the list, in the changes document |
| 3. Port the changes behind the version switch | edit `py_impl.py`, re-assemble; add the release to `VERSION.json` | per-function tests pinned from the new release run under Octave ([`VERIFICATION.md`](VERIFICATION.md)) |
| 4. Regenerate the Octave release file | `octave/make_octave_compat.py` against the new `matlab/` file | `tests/test_octave_compat.py` green |
| 5. Rerun the benchmark: existing cases first | the reference cases on the new release before anything else, so the delta between releases is measured, not assumed | a `MATLAB_<ver>_IMPACT.md`, as [`MATLAB_4p16p0_IMPACT.md`](MATLAB_4p16p0_IMPACT.md) |
| 6. Publish the correlation tables | agreement statistics only; reference values are never published here | [`VERSIONS.md`](VERSIONS.md) and [`../MATLAB_Correlation_Review.md`](../MATLAB_Correlation_Review.md) updated |
| 7. Make it the default, and release | `VERSION.json` default, CHANGELOG, tag (§5) | the tag |

**Which builds stay selectable.** An older build stays selectable only while
published correlation evidence depends on it. Today that is the 4p15p0 build
with the adaptive local search, which produced the 208-case reference results.
When no published result needs a build any more, it can be retired.

## 3. Between releases

- **A port defect** is fixed with a test that fails without the fix. If it can
  move a COM, FOM or sampling-phase value, rerun the reference cases before
  committing it and add an entry to [`FIX_SUMMARY.md`](FIX_SUMMARY.md).
- **A speed-up** lands only if `tools/equivalence_check.py` passes on all 28
  checkpoint cases. Run `tests/test_mutation_score.py` before any engine commit.
- **Nothing else changes the engine.** In particular, nothing improves on the
  Reference Code, not even behind an opt-in switch. An improvement is a proposal
  for the ad hoc.

The gate before every push is `tests/run_all.ps1`. After it, check CI with
`gh run list --limit 3`.

## 4. Issues, and pull requests

Issues are how outside help arrives. Triage each against the table in
CONTRIBUTING.md, *What happens to it*:

| an issue that is | answer |
|---|---|
| a port defect | reproduce, fix, and say in the thread when the reference rerun will be done, rather than leaving it silent |
| Reference Code behaviour | explain, with the MATLAB lines. A real Reference Code defect is reported to the ad hoc and tracked there |
| a feature or method request | point to the COM ad hoc |
| about a supporting tool | no commitment. Fix it if it is worth it |
| a request for channel files | they are not yours to give. Point to CONTRIBUTING.md, *Correlation data*, which links the IEEE 802.3dj page that lists the contributions by name |
| a security report | follow [`../SECURITY.md`](../SECURITY.md) |

**Pull requests are not accepted**, and they are turned off in the repository
settings, so none can be opened. If they are ever turned back on, close one with
a pointer to the issue tracker. If an idea in an issue is right, write the change yourself from the description,
not by merging or copying the submitted code, and credit the reporter in the
commit message. Every line in the repository then comes from the maintainer or
the Reference Code, which is what keeps the licence record simple without a
sign-off. A pull request can carry CI changes that run on this repository's
runners. Do not run a fork's workflow without reading it.

CI still has a `dco` job. It runs only on pull requests, so with none accepted
it never fires. It is left in place rather than removed, in case the policy
changes.

**Repository settings.** Issues on, pull requests off, wiki off (the
documentation is versioned with the code), Projects off, discussions off unless
the ad hoc asks for a place to talk that is not the reflector. A ruleset on
`master` blocks force-pushes and deletion. Secret scanning with push protection,
Dependabot alerts and private vulnerability reporting are on. Nobody else has write access. Granting it is a decision
about the commitment, not a reward for a good report.

## 5. Releasing

`master` is what users get. Mark each state people may depend on with a tag,
which is a permanent name for a commit and what a citation needs:

```bash
git tag -a v1.0.0 -m "..."
git push origin v1.0.0
```

For each release, update these together:

- `CHANGELOG.md`;
- the version in `pyproject.toml` and in `CITATION.cff`, with its
  `date-released`;
- `VERSION.json` if the emulated Reference Code release changed.

Prefer a tag over a commit SHA in anything written down: SHAs change if history
is rewritten, and this repository's history has been rewritten once already, to
purge data. While the repository is private a tag can still be re-cut; once
public, a published tag should not move.
