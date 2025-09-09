# GCSU Nuclear Physics Toolkit


A modern Python implementation of the legacy Fortran STOPX and KINEQ codes developed at Oak Ridge National Laboratory. This toolkit provides fast energy loss calculations and nuclear reaction kinematics for charged particles in matter.

## Overview

The GCSU Nuclear Physics Toolkit maintains exact numerical fidelity to the original Fortran implementations while providing modern Python interfaces suitable for batch calculations and integration with the scientific Python ecosystem.

### Key Features

- **Exact Translation**: All algorithms, constants, and numerical tolerances preserved from original Fortran codes
- **High Performance**: JIT compilation with Numba for core calculations
- **Vectorized Interfaces**: Batch processing capabilities for large-scale calculations
- **Modern Design**: Object-oriented Python interface with comprehensive documentation
- **Comprehensive Testing**: Extensive validation against original codes

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- Numba
- Matplotlib (for plotting examples)

### Installation

Clone the repository and install in development mode:

```bash
git clone <repository-url>
cd gcsu_nuclear
pip install -e .
```

Or install dependencies manually:

```bash
pip install numpy numba matplotlib
```

## Quick Start

### Energy Loss Calculations

```python
from src.stopx_model import StopxModel

# Initialize the model
stopx = StopxModel()

# Define projectile: 51V (Z=23, A=51)
stopx.define_projectiles([(23, 51)])

# Define absorber: Silver (Z=47, A=107.87)
stopx.define_absorbers([{
    'composition': [(47, 107.87, 1.0)],  # Pure silver
    'thickness': 1.0  # 1.0 mg/cm²
}])

# Calculate stopping power at 150 MeV
stopping_power = stopx.calculate_stopping_power(150.0, 0, 0)
print(f"Stopping power: {stopping_power:.2f} MeV/(mg/cm²)")

# Calculate energy loss
energy_loss = stopping_power * 1.0  # thickness = 1.0 mg/cm²
final_energy = 150.0 - energy_loss
print(f"Energy loss: {energy_loss:.2f} MeV")
print(f"Final energy: {final_energy:.2f} MeV")
```

### Reaction Kinematics

```python
from src.kineq_model import KineqModel

# Initialize the model
kineq = KineqModel()

# Parse reaction: 12C(4He,1n)15N
success = kineq.parse_reaction("12C(4He,1n)15N")
if success:
    print(f"Q-value: {kineq.q_value:.3f} MeV")
    
    # Set up angle calculation
    kineq.set_angle_data(elab=50.0, amin=0.0, amax=180.0, da=10.0)
    
    # Calculate kinematics
    results = kineq.calculate_kinematics()
    print(f"Number of angles calculated: {len(results['angles'])}")
```

## Examples

Run the provided examples:

```bash
python examples.py
```

This will demonstrate:
1. 51V ion (150 MeV) through 1.0 mg/cm² of silver
2. 197Ag ion (200 MeV) through 3.0 mg/cm² of nickel
3. Energy loss curves and range calculations
4. Reaction kinematics examples

## Theory and Background

### Energy Loss Theory

The energy loss of charged particles in matter is described by the Bethe-Bloch equation:

$$-\frac{dE}{dx} = \frac{4\pi z^2 e^4}{m_e v^2} n Z \left[ \ln\left(\frac{2m_e v^2}{I}\right) - \ln(1-\beta^2) - \beta^2 \right]$$

Where:
- $z$ is the projectile charge
- $v$ is the projectile velocity
- $n$ is the electron density
- $Z$ is the target atomic number
- $I$ is the mean ionization potential
- $\beta = v/c$ is the relativistic factor

### Nuclear Stopping Power

At low energies, nuclear stopping becomes important:

$$S_n = \frac{4\pi a^2 Z_1 Z_2 e^2 \gamma \lambda}{A_1}$$

Where $\gamma$ and $\lambda$ are dimensionless functions of the reduced energy.

### Reaction Kinematics

For a reaction $A(a,b)B$, the Q-value is:

$$Q = (m_A + m_a - m_b - m_B)c^2$$

The threshold energy for the reaction is:

$$E_{th} = -Q \frac{m_A + m_a + m_b + m_B}{2m_A}$$

## Historical Context

The STOPX and KINEQ codes were developed at Oak Ridge National Laboratory in the 1970s and 1980s. They represent some of the most accurate and widely-used codes for nuclear physics calculations. This Python implementation maintains exact numerical fidelity to the original Fortran codes while providing modern interfaces and performance optimizations.

### Original Fortran Codes

- **STOPX**: Energy loss calculations for charged particles in matter
- **KINEQ**: Nuclear reaction kinematics calculations

Both codes were extensively validated and used in nuclear physics research for decades.

## API Reference

### StopxModel

The `StopxModel` class provides energy loss calculations:

#### Methods

- `define_projectiles(projectiles)`: Define projectile particles
- `define_absorbers(absorbers)`: Define absorber materials
- `calculate_stopping_power(energy, projectile_idx, absorber_idx)`: Calculate stopping power
- `calculate_range(start_energy, end_energy, step_size, projectile_idx, absorber_idx)`: Calculate particle range
- `calculate_energy_loss(incident_energies, projectile_idx)`: Calculate energy loss through absorbers

### KineqModel

The `KineqModel` class provides reaction kinematics:

#### Methods

- `parse_reaction(reaction_string)`: Parse nuclear reaction specification
- `set_mode(mode)`: Set calculation mode (REL, CLAS, or Q)
- `set_angle_data(elab, amin, amax, da)`: Set angle calculation parameters
- `set_energy_data(alab, emin, emax, de)`: Set energy calculation parameters
- `calculate_kinematics(angles, energies)`: Calculate reaction kinematics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Ziegler, J.F., Biersack, J.P., and Littmark, U. "The Stopping and Range of Ions in Solids." Pergamon Press, 1985.
