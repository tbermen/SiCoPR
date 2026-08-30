"""Launch `sicopr.py` from the GUI and stream its output back.

One run at a time. This is a single-user local tool and a COM run is CPU-heavy;
letting the browser start several by clicking twice would be a way to bring the
machine to its knees, not a feature.

**No shell, ever.** The command is built as an argv list and handed to
subprocess directly, and every path in it is validated by the caller before it
gets here. A GUI that shells out with user-supplied strings is a command
injection waiting to happen, even bound to localhost.

`PYTHONUNBUFFERED` is set on the child because otherwise Python block-buffers
stdout when it is a pipe, and the status window would sit empty for minutes and
then dump everything at once -- which defeats the point of having one.
"""
import collections
import os
import subprocess
import sys
import threading
import time

_MAX_LINES = 20000        # a long corpus run must not grow the buffer forever

# Most lines one status response may carry. A full-grid search is not just long,
# it is loud, and without a cap a single poll could hand the browser the whole
# 20 000-line buffer (~1 MB of JSON) and then do it again a second later. The
# client keeps its offset and drains the backlog over successive polls.
_MAX_CHUNK = 2000


class Run:
    """A single sicopr.py invocation and everything it has printed so far."""

    def __init__(self, argv, cwd, label=''):
        self.argv = list(argv)
        self.cwd = cwd
        self.label = label
        # deque with maxlen: appending past the cap drops the oldest in O(1).
        # A plain list needed `del lines[:1]` per line once full, which moves
        # every remaining element and gets slower exactly when output is
        # heaviest.
        self.lines = collections.deque(maxlen=_MAX_LINES)
        self.started = time.time()
        self.finished = None
        self.returncode = None
        self.truncated = 0
        self.last_output = self.started
        self._proc = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        self._proc = subprocess.Popen(
            self.argv, cwd=self.cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True, bufsize=1, errors='replace')
        threading.Thread(target=self._pump, daemon=True).start()
        return self

    def _pump(self):
        try:
            for line in self._proc.stdout:
                self._append(line.rstrip('\n'))
        except Exception as e:                        # noqa: BLE001
            self._append('[gui] error reading output: %s' % e)
        finally:
            self.returncode = self._proc.wait()
            self.finished = time.time()
            self._append('[gui] exited with code %s after %.1fs'
                         % (self.returncode, self.finished - self.started))

    def _append(self, text):
        with self._lock:
            if len(self.lines) == _MAX_LINES:
                self.truncated += 1      # deque is about to evict the oldest
            self.lines.append(text)
            self.last_output = time.time()

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._append('[gui] stop requested')
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._append('[gui] did not exit; killing')
                self._proc.kill()
            return True
        return False

    # -- reporting ---------------------------------------------------------
    @property
    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def status(self, since=0):
        """Lines from `since` onward, plus where the run is up to.

        `since` counts lines ever produced, not lines currently buffered, so a
        client that falls behind a truncating run is told to resync rather than
        silently shown the wrong window of output.
        """
        with self._lock:
            if since < self.truncated:
                start = 0
                resync = True
            else:
                start = since - self.truncated
                resync = False
            buf = list(self.lines)
            chunk = buf[start:start + _MAX_CHUNK]
            behind = max(0, len(buf) - start - len(chunk))
            quiet = time.time() - self.last_output
            # How far the client has actually consumed -- NOT the total. With a
            # capped chunk those differ, and returning the total would make the
            # client skip every line it was not sent.
            consumed = self.truncated + start + len(chunk)
        return {
            'behind': behind,
            'quiet': quiet,
            'running': self.running,
            'returncode': self.returncode,
            'argv': self.argv,
            'label': self.label,
            'started': self.started,
            'elapsed': (self.finished or time.time()) - self.started,
            'lines': chunk,
            'next': consumed,
            'resync': resync,
            'truncated': self.truncated,
        }


_current = None
_guard = threading.Lock()


def current():
    return _current


def start(argv, cwd, label=''):
    """Start a run, refusing if one is already going."""
    global _current
    with _guard:
        if _current is not None and _current.running:
            raise RuntimeError(
                'a run is already in progress (%s); stop it first'
                % ' '.join(_current.argv[:3]))
        _current = Run(argv, cwd, label).start()
        return _current


def stop():
    with _guard:
        return bool(_current and _current.stop())
