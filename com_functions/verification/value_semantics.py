"""MATLAB value semantics, as checks a function test can make.

MATLAB copies on assignment: `ch.sdd21 = ch.sdd21_raw` gives two arrays, and
writing one never changes the other. numpy assigns by reference, so the port
writes `.copy()` wherever the reference's later code could tell the difference,
and dropping one is the `drop_dot_copy` mutation (the class behind the
`BEST.PSD_results` and `process_sxp` OP leaks). A value test cannot see a
dropped copy until some later code writes in place, which is why 31 functions'
own tests let it through.

These two checks see it directly:

    assert_no_shared_memory(chdata=out)    no two arrays reachable from the
                                           outputs share memory
    snap = snapshot(OP=OP); ...; assert_unchanged(snap, OP=OP)
                                           an input the reference treats as
                                           read-only comes back unchanged

They test a property of the reference (MATLAB cannot alias two variables),
not a number, so they need no oracle.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import copy
import itertools

import numpy as np


def arrays(**roots):
    """[(path, ndarray)] for every non-empty array reachable from the roots,
    through attributes, dict values and list/tuple items. Each container is
    walked once, so one object reached twice is not a finding; one ARRAY held
    in two places is."""
    out, seen = [], set()

    def walk(obj, path):
        if isinstance(obj, np.ndarray):
            if obj.size and obj.dtype != object:
                out.append((path, obj))
            elif obj.dtype == object:
                for i, v in enumerate(obj.flat):
                    walk(v, '%s.flat[%d]' % (path, i))
            return
        if id(obj) in seen:
            return
        if isinstance(obj, dict):
            seen.add(id(obj))
            for k, v in obj.items():
                walk(v, '%s[%r]' % (path, k))
        elif isinstance(obj, (list, tuple)):
            seen.add(id(obj))
            for i, v in enumerate(obj):
                walk(v, '%s[%d]' % (path, i))
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            seen.add(id(obj))
            for k, v in vars(obj).items():
                walk(v, '%s.%s' % (path, k))

    for name, obj in roots.items():
        walk(obj, name)
    return out


def shared_memory(**roots):
    """Pairs of paths whose arrays overlap in memory."""
    found = arrays(**roots)
    return [(p, q) for (p, a), (q, b) in itertools.combinations(found, 2)
            if np.may_share_memory(a, b) and np.shares_memory(a, b)]


def assert_no_shared_memory(**roots):
    pairs = shared_memory(**roots)
    assert not pairs, ('arrays that MATLAB would hold as separate values share '
                       'memory: %s' % ', '.join('%s ~ %s' % pq for pq in pairs[:8]))


def snapshot(**roots):
    return copy.deepcopy(roots)


def assert_unchanged(snap, **roots):
    """Every array reachable from each root is bitwise what it was."""
    before = dict(arrays(**snap))
    after = dict(arrays(**roots))
    assert set(before) == set(after), 'arrays added or removed: %s' % sorted(
        set(before) ^ set(after))[:8]
    changed = [p for p in before
               if before[p].shape != after[p].shape
               or before[p].tobytes() != after[p].tobytes()]
    assert not changed, 'input changed in place: %s' % changed[:8]
