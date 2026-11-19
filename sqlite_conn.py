"""
sqlite3 connections that are always closed.

`with sqlite3.connect(path) as con:` commits (or rolls back) but does NOT close
the connection; it is only closed later by the garbage collector. For a bot that
runs 24/7 (and on Windows, where an open handle keeps the file locked) every
connection is closed as soon as its block ends:

    from sqlite_conn import connect
    with connect(path) as con:      # commit/rollback as before, then close
        ...
"""
import sqlite3
from contextlib import contextmanager


@contextmanager
def connect(path, **kwargs):
    con = sqlite3.connect(path, **kwargs)
    try:
        with con:              # same commit / rollback as `with sqlite3.connect(...)`
            yield con
    finally:
        con.close()
