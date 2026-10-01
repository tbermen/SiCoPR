# Security policy

## What this software is, in security terms

SiCoPR is an offline numerical tool. It reads Touchstone S-parameter files and
an Excel configuration spreadsheet, computes Channel Operating Margin, and
writes figures and result files. It opens no network connections, serves no
requests and stores no credentials.

One component is different and worth knowing about: the optional configuration
editor under `gui/` runs a local HTTP server, reads and writes files by path,
and **spawns `sicopr.py` as a subprocess**. It binds to `127.0.0.1` only. Every
path parameter is checked to be inside one of the roots the user gave it — the
repository, any directory passed with `--dir`/`--run-dir` or `SICOPR_DIRS`, and
any directory opened in the session — file types are restricted, and commands
are built as argument lists without a shell. It is
built for a single local user and should not be exposed to a network.

## Reporting a vulnerability

Open a GitHub issue at https://github.com/tbermen/SiCoPR/issues.

If the problem is one you would rather not describe publicly, report it
privately instead: **Security** tab, then **Report a vulnerability**
(https://github.com/tbermen/SiCoPR/security/advisories/new). The report is
not public.

There is no bug bounty, and no service-level commitment on response time — this
is a single-maintainer project. Reports are read and acted on.

## What is in scope

- Anything in the `gui/` server that reads, writes or executes outside its
  intended bounds: a path that escapes the repository, a file type that should
  not be served, a way to influence the spawned command.
- Code execution reachable by opening a crafted configuration workbook or
  Touchstone file with the engine.
- A dependency vulnerability that this project's usage actually exposes.

## What is not

- **Numerical disagreement with the MATLAB reference is not a security issue.**
  It is a correctness issue and belongs in a normal issue report; see
  `CONTRIBUTING.md`.
- Exposing the `gui/` server to a network and finding it reachable. It binds to
  localhost by design and is documented as a single-user local tool.
- Denial of service by handing the engine a very large channel file. COM runs
  are expected to be long and memory-hungry.

## Supported versions

`master` is the release; there are no maintained release branches. Fixes land on
`master`.
