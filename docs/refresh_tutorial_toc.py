"""Rewrite the Contents block of docs/TUTORIAL.md from its own headings.

    python docs/refresh_tutorial_toc.py [--check]

A hand-maintained table of contents in an 11,000-word document goes stale the
first time a section is added. This regenerates it, and `--check` fails if the
committed file is not what it produces, which is how the test keeps it honest.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import io
import os
import re
import sys

DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TUTORIAL.md')
START = '## Contents'


def anchor(heading):
    """GitHub's rule: lowercase, drop punctuation, one hyphen per space."""
    a = heading.lower().replace('`', '')
    a = re.sub(r'[^\w\s-]', '', a)
    return a.strip().replace(' ', '-')


def headings(text):
    out, in_fence = [], False
    for line in text.split('\n'):
        if line.startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r'^(#{1,3}) (.+)$', line)
        if m and m.group(2).strip() != 'Contents':
            out.append((len(m.group(1)), m.group(2).strip()))
    return out


def build(text):
    body_start = text.index(START)
    after = text.index('\n# ', body_start)          # first chapter heading
    toc = ['%s- [%s](#%s)' % ('  ' * (lvl - 1), h, anchor(h))
           for lvl, h in headings(text[after:])]
    return text[:body_start] + START + '\n\n' + '\n'.join(toc) + '\n' + text[after:]


def main(argv):
    text = io.open(DOC, encoding='utf-8').read()
    fresh = build(text)
    if '--check' in argv:
        if fresh != text:
            print('docs/TUTORIAL.md: the Contents block is out of date; '
                  'run python docs/refresh_tutorial_toc.py')
            return 1
        print('docs/TUTORIAL.md: contents up to date (%d entries)'
              % len(headings(text[text.index('\n# ', text.index(START)):])))
        return 0
    io.open(DOC, 'w', encoding='utf-8', newline='\n').write(fresh)
    print('rewrote the Contents block: %d entries' % len(headings(text)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
