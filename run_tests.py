"""Portable one-command test runner: python run_tests.py"""
import sys
import pytest
if __name__=='__main__':
    raise SystemExit(pytest.main(['tests','-q',*sys.argv[1:]]))
