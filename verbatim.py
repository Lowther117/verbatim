#!/usr/bin/env python3
"""Verbatim - audio and video to text, with speaker labels, entirely offline.

  python verbatim.py                       open the window
  python verbatim.py FILE ...              open the window with files queued
  python verbatim.py transcribe FILE ...   transcribe from the command line (--help for options)
  python verbatim.py models                list the models and which are installed
  python verbatim.py fetch ID ...          download models
  python verbatim.py selftest              check the whole pipeline end to end
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verbatimlib.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
