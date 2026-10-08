"""
Tests whole benchmarking pipeline in main.py
"""

import pytest 

from benchmark.main import main


class args:
    def __init__(self):
        self.optim = False
        self.saved = "original_params.pkl"
        self.paramsf = "pytest_output.pkl"
        self.resultsf = "pytest_ouptut.pkl"
        self.testf = "pytest_ouptut.pkl"
        self.epochs = 1
        self.repeats = 1
        self.n = 0.01
        self.B = 1


class Test_main:

    def test_main_executes(self):
        assert main(args()) == 0
