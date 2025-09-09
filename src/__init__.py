"""
GCSU Nuclear Physics Toolkit: Python Modernization of KINEQ and STOPX

This package provides modern Python implementations of the legacy Fortran 77
codes KINEQ (reaction kinematics) and STOPX (energy loss calculations) developed
at Oak Ridge National Laboratory.

The implementation maintains absolute numerical fidelity to the original codes
while providing a modern, high-performance interface suitable for batch
calculations and integration with the scientific Python ecosystem.

Key Features:
- Exact translation of all physical models and numerical algorithms
- High-performance JIT compilation for core calculations
- Vectorized interfaces for batch processing
- Comprehensive test suite for validation
- Modern object-oriented design

Classes:
    StopxModel: Energy loss calculations in matter
    KineqModel: Nuclear reaction kinematics calculations

Author: Joel Saucedo
"""

from .stopx_model import StopxModel
from .kineq_model import KineqModel

__version__ = "1.0.0"
__author__ = "GCSU Nuclear Physics"
__all__ = ["StopxModel", "KineqModel"]
