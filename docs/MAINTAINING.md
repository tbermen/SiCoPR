# Maintaining this repository

The other side of [`CONTRIBUTING.md`](../CONTRIBUTING.md): what the maintainer
does, and how the repository is configured so that the rules there are enforced
by the platform rather than by memory.

Written for someone who has not run a public repository before. None of it is
elaborate — the whole point is that almost nothing should depend on remembering
to do the right thing.

---

## 1. Who can change what

Making a repository public grants the world **read** access. It does not grant
anyone write access. Strangers cannot push to it, cannot delete it, and cannot
change a line of it.

What they can do is **fork** it — take their own copy — and then open a **pull
request**, which is a request that you merge their copy's changes into yours.
You decide. Nothing lands without you.

| who | can read | can push | can merge |
|---|---|---|---|
| anyone on the internet | yes | no | no |
| a contributor with a fork | yes | to their own fork only | no |
| a collaborator you invite | yes | yes, if you give write access | yes, if you allow it |
| you | yes | yes | yes |

So the answer to "will others be able to push changes?" is **no, not unless you
invite them by name.** The default is exactly what you want.

## 2. Settings to apply once

On GitHub, *Settings → Branches → Add branch ruleset* for `master`:

| setting | why |
|---|---|
| **Require a pull request before merging** | stops anyone — you included — pushing straight to `master`. This is the one that matters most: it means every change has a diff someone looked at. |
| **Require status checks to pass** → select the `suite`, `licence` and `hygiene` jobs | a red CI cannot be merged. The `hygiene` job is what stops correlation data and MATLAB-derived values being re-added; see [`MIN_RADIUS_ASSUMPTION.md`](MIN_RADIUS_ASSUMPTION.md) for the kind of mistake CI is there to catch. |
| **Require branches to be up to date before merging** | the checks ran against what will actually be on `master`, not against a stale base. |
| **Do not** require approvals from others | you are the only maintainer; requiring a second reviewer would block you entirely. Revisit if that changes. |

Also under *Settings → General*:

- **Issues: on.** This is how defects arrive from people who cannot or will not
  write the fix. It is the cheapest signal you will get.
- **Discussions: optional.** Useful if the ad hoc wants a place to talk that is
  not the reflector. Leave it off until someone asks.
- **Wiki: off.** The documentation is in the repository, where it is versioned
  and reviewed with the code. A wiki is a second place for it to go stale.

You can approve and merge your own pull requests. That is normal for a
single-maintainer project and is not a loophole — the value is the diff and the
green CI, not a second signature.

## 3. Reviewing a pull request from a stranger

Read the diff before you run anything. A pull request can change CI
configuration, add a dependency, or add a script — and CI on a fork's pull
request runs code the author wrote. Specifically check:

- **Does it touch `.github/workflows/`?** Treat that as a change to what runs on
  your machine and read it line by line.
- **Does it add a dependency?** A new import in `requirements.txt` is a new piece
  of software you are asking every user to install. Ask what it buys.
- **Does it edit `com.py` directly?** Reject it — `com.py` is generated, and CI
  catches this, but say why so the contributor knows to redo it under
  `com_functions/fn/`.
- **Does it add data?** Channel files, configuration workbooks and MATLAB
  reference values do not belong here, whoever they came from. The `hygiene` job
  catches the known paths; a new path is your judgement.
- **Can it move a number?** If yes, the correlation set has to be re-run before
  merge, and only you can do that — the data is not public. Say so in the thread
  and give a rough timescale rather than leaving it silent.

## 4. Releasing

There is no release process, and none is needed yet: `master` is the release, and
`VERSION.json` records which MATLAB release the engine emulates.

If that changes — if people start depending on a specific state — tag it:

```bash
git tag -a v1.0 -m "first public release, 208/208 against the 4p15p0 reference"
git push origin v1.0
```

A tag is a permanent name for a commit, which is what a citation needs. Prefer a
tag over a commit SHA in anything written down: SHAs change if history is ever
rewritten, and this repository's history has been rewritten once already, to
purge data.

## 5. Things that would need a decision, not a commit

- **Someone asks to be a collaborator.** Write access is not something to grant
  because a contribution was good; grant it when you want that person to be able
  to merge *without you*. Until then, their pull requests are the mechanism.
- **Someone proposes a change that improves on the MATLAB.** The project's rule
  is fidelity over improvement. That is worth explaining rather than just
  declining — and worth reconsidering only as an opt-in switch, never as a
  default.
- **A vendor asks for the channel files.** They are not yours to give. Point at
  README §1, which names the IEEE contributions each one comes from.
- **Someone reports a security problem.** There is no attack surface to speak of
  — this reads local files and does arithmetic — but if one is reported, ask them
  to email rather than open a public issue, and add a `SECURITY.md` saying so if
  it happens more than once.
