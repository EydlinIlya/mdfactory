# ABOUTME: Test helper utilities providing a multiprocessing.Process subclass
# ABOUTME: that captures and re-raises exceptions from child processes.
"""Test helper utilities providing a multiprocessing.Process subclass."""

import multiprocessing
import traceback


class ProcessWithException(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
