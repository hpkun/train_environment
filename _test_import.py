"""Minimal test: does importing my_uav_env break stdout?"""
import os, sys

# Write diagnostic to file (survives any stdout issue)
_DIAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_import.txt")
def _mark(msg):
    try:
        with open(_DIAG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

_mark("test_start")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_mark("kmp_set")

# Test 1: basic print
print("=== Test 1: This should print BEFORE my_uav_env import ===", flush=True)
sys.stdout.flush()
_mark("test1_print_done")

import numpy as np
print("=== Test 2: numpy imported ===", flush=True)
sys.stdout.flush()
_mark("test2_numpy_done")

import torch
print("=== Test 3: torch imported ===", flush=True)
sys.stdout.flush()
_mark("test3_torch_done")

# This is the critical import that triggers simulator.py → import jsbsim
print("=== Test 4: importing my_uav_env... ===", flush=True)
sys.stdout.flush()
_mark("test4_before_uav_import")

from my_uav_env import UavCombatEnv
_mark("test4_after_uav_import")
print("=== Test 5: my_uav_env imported OK ===", flush=True)
sys.stdout.flush()
_mark("test5_done")

print("=== ALL TESTS PASSED ===", flush=True)
_mark("all_done")
