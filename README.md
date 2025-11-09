# Polydisperse Diffusiophoretic Colloid Simulation

Python translation of Fortran code for fast, particle-resolved simulations of polydisperse hard-sphere colloids in a 2D domain under reaction-diffusion patterns.

**Original Fortran code by:** Siamak Mirfendereski  
**Python translation by:** Claude (Anthropic)

---

## Overview

This simulation package implements:

1. **Reaction-Diffusion Solver** - Solves Brusselator model equations to generate Turing patterns
2. **Advection-Diffusion Solver** - Computes continuum colloidal concentration fields
3. **Particle Dynamics** - Lagrangian tracking of polydisperse hard spheres with:
   - Diffusiophoresis (motion driven by concentration gradients)
   - Brownian dynamics (thermal fluctuations)
   - Hard-sphere collision detection and resolution

## Physical Model

### Brusselator Reaction-Diffusion System

The morphogen concentrations X (C1) and Y (C2) evolve according to:

```
∂X/∂t = D_X ∇²X + Da_c(A - (B+1)X + X²Y)
∂Y/∂t = D_Y ∇²Y + Da_c(BX - X²Y)
```

where:
- A, B are control parameters
- Da_c is the Damköhler number
- D_Y = D_rd × D_X (relative diffusion coefficient)

### Particle Dynamics

Each particle experiences:

1. **Diffusiophoretic velocity**: `V_dp = Mob_C1 × ∇C1 + Mob_C2 × ∇C2`
2. **Brownian velocity**: Random thermal motion scaled by particle size
3. **Hard-sphere collisions**: Resolved using fast subcell algorithm

Position update (Adams-Bashforth):
```
x^(n+1) = x^n + (3/2)V^n - (1/2)V^(n-1)) Δt
```

### Continuum Advection-Diffusion

Colloidal volume fractions φ₁ and φ₂ satisfy:

```
∂φ/∂t + ∇·(V_dp φ) = D_eff ∇²φ
```

---

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib numba
```

### Optional (for animations)
```bash
pip install ffmpeg-python
```

### Files Structure

```
.
├── rd_dpps_simulation.py      # Main simulation class
├── reaction_diffusion.py       # Brusselator solver
├── advection_diffusion.py      # Continuum solver
├── particle_dynamics.py        # Collision detection
├── velocity_calculator.py      # Force/velocity computation
├── utils.py                    # Utilities and visualization
├── example_simple.py           # Quick start example
└── README.md                   # This file
```

---

## Quick Start

### Minimal Example

```python
from rd_dpps_simulation import PolyDispDPBrownianSimulation

# Create simulation with reduced parameters for quick test
sim = PolyDispDPBrownianSimulation(
    N=1000,              # Total particles
    N1=500,              # Type 1 particles
    nx_rd=128,           # RD grid size
    ny_rd=128,
    max_step_rd=10000,   # RD time steps
    maxc_step=5000,      # AD time steps
    t_final=50.0,        # Simulation time
    delta_t=0.05,        # Time step
    output_dir="test_output"
)

# Run complete simulation
sim.run()
```

### Production Run

```python
# Full-scale simulation (as in original Fortran)
sim = PolyDispDPBrownianSimulation(
    N=180000,
    N1=90000,
    nx_rd=640,
    ny_rd=640,
    max_step_rd=6800000,
    maxc_step=3200000,
    t_final=2000.0,
    Pe=20.0,
    A_rd=4.5,
    mu=0.04,
    D_rd=8.0,
    output_dir="production_output"
)

sim.run()
```

---

## Key Parameters

### Particle Parameters
- `N`: Total number of particles (e.g., 180000)
- `N1`: Number of type-1 particles (e.g., 90000)
- `mid_p_size`, `low_p_size`: Size distribution parameters (polydisperse)

### Domain Parameters
- `size_x`, `size_y`: Reaction-diffusion domain size (e.g., 32.0)
- `box_scale`: Physical domain scaling (e.g., 53.08)
- `nx_rd`, `ny_rd`: Grid resolution (e.g., 640)

### Time Parameters
- `delta_t`: Particle dynamics time step (e.g., 0.05)
- `dt_rd`: Reaction-diffusion time step (e.g., 0.00003)
- `dt_ad`: Advection-diffusion time step (e.g., 0.000005)
- `t_final`: Total simulation time (e.g., 2000.0)
- `max_step_rd`: RD solver iterations (e.g., 6800000)
- `maxc_step`: AD solver iterations (e.g., 3200000)

### Physical Parameters
- `Pe`: Péclet number (ratio of advection to diffusion) (e.g., 20.0)
- `A_rd`: Brusselator parameter A (e.g., 4.5)
- `mu`: Control parameter μ (e.g., 0.04)
- `D_rd`: Relative diffusion coefficient (e.g., 8.0)
- `Da_c`: Damköhler number (e.g., 1.0)

### Mobility Parameters
- `Mob_C1`, `Mob_C2`: Type-1 mobilities w.r.t. C1 and C2 (e.g., 0.1, -0.1)
- `Mob2_C1`, `Mob2_C2`: Type-2 mobilities (e.g., -0.1, 0.1)

---

## Output Files

### Particle Data
- `pos####.txt`: Particle positions at each save interval
- `vel####.txt`: Particle velocities at each save interval
- `particle_radius.txt`: Particle sizes (polydisperse)

### Field Data
- `Brusselator_SS.txt`: Steady-state morphogen concentrations (C1, C2)
- `grad_Brus_SS.txt`: Concentration gradients (∇C1, ∇C2)
- `colloids_vf.txt`: Colloidal volume fractions (φ₁, φ₂)

### Statistics
- `ms.txt`: Mean-square displacements
- `autocorrelation_vel.txt`: Velocity autocorrelation functions
- `par1.txt`: Trajectory of particle #1

### Visualizations (if generated)
- `turing_pattern.png`: Brusselator steady state
- `particles.png`: Particle configuration
- `msd.png`: Mean-square displacement plot
- `vacf.png`: Velocity autocorrelation plot
- `trajectory.png`: Single particle trajectory
- `animation.mp4`: Particle motion animation

---

## Usage Examples

### 1. Solve Only Reaction-Diffusion

```python
from reaction_diffusion import ReactionDiffusionSolver, visualize_turing_pattern

solver = ReactionDiffusionSolver(
    nx=640, ny=640,
    size_x=32.0, size_y=32.0,
    dt=0.00003, max_steps=1000000,
    tol=1e-12,
    A=4.5, D_rd=8.0, mu=0.04,
    Da_c=1.0, am_noise=0.02
)

C1, C2, Dx_C1, Dx_C2, Dy_C1, Dy_C2, alpha = solver.solve()

visualize_turing_pattern(C1, C2, "turing.png")
```

### 2. Resume from Checkpoint

```python
# Simulation automatically detects and loads checkpoints
sim = PolyDispDPBrownianSimulation(
    output_dir="existing_output"  # Contains pos####.txt files
)

# Will resume from last saved state
sim.run_particle_dynamics()
```

### 3. Post-Processing

```python
from utils import analyze_results

analyze_results(Path("output"))
# Generates: msd.png, vacf.png, trajectory.png
```

### 4. Create Animation

```python
from utils import create_animation
from pathlib import Path
import numpy as np

box = np.array([1700, 1700, 2.5])
size_particle = np.ones(1000)  # Or load from particle_radius.txt

create_animation(
    output_dir=Path("output"),
    box=box,
    N1=500,
    size_particle=size_particle,
    output_file="animation.mp4",
    max_frames=100
)
```

---

## Performance Optimization

### For Large Simulations

1. **Use NumPy efficiently**: Vectorized operations are already implemented
2. **Numba JIT compilation**: The particle dynamics module uses Numba for speed
3. **Reduce output frequency**: Set `save_interval` larger (e.g., 100 instead of 60)
4. **GPU acceleration**: Could be added using CuPy (not currently implemented)

### Typical Runtimes (approximate)

| Configuration | Grid | Particles | Time Steps | Runtime |
|---------------|------|-----------|------------|---------|
| Quick test | 128² | 1,000 | 1,000 | ~10 min |
| Medium | 320² | 10,000 | 10,000 | ~2 hours |
| Full scale | 640² | 180,000 | 40,000 | ~days |

*Runtimes depend heavily on CPU and collision frequency*

---

## Algorithm Details

### Collision Detection

Fast subcell grid algorithm:
1. Domain divided into subcells (~9 units)
2. Particles assigned to overlapping subcells
3. Only nearby particles checked for collision
4. Complexity: O(N) instead of O(N²)

### Time Integration

- **Reaction-Diffusion**: Adams-Bashforth 2nd order
- **Advection-Diffusion**: Adams-Bashforth with upwind advection
- **Particle Dynamics**: Adams-Bashforth with periodic boundaries

### Numerical Stability

- **CFL condition**: Automatically satisfied with chosen time steps
- **Convergence**: Both RD and AD solvers monitor residuals
- **Conservation**: Mass conservation verified at each output

---

## Differences from Original Fortran

1. **Object-oriented design**: Modular class structure
2. **Vectorization**: NumPy operations replace explicit loops where possible
3. **Simplified I/O**: Direct NumPy text format, optional HDF5
4. **Added visualization**: Built-in plotting and animation tools
5. **Removed dependencies**: No external FFT or exact solvers needed for this version
6. **Type hints**: Python 3.10+ style type annotations

---

## Troubleshooting

### Memory Issues
- Reduce `N` (number of particles)
- Reduce grid resolution (`nx_rd`, `ny_rd`)
- Increase `sub_g_size` to use fewer subcells

### Slow Performance
- Reduce `max_step_rd` and `maxc_step` for quicker convergence check
- Increase time steps (`dt_rd`, `dt_ad`) carefully
- Use smaller `t_final` for testing

### Collision Errors
- Particles still overlap: Increase `max_iter` in collision resolution
- Try adjusting `distance_cutoff` parameter
- Check initial particle placement (`assign_particle_positions`)

### Numerical Instability
- Reduce time steps if solution diverges
- Check Péclet number isn't too large
- Verify boundary conditions are correct

---

## Citation

If you use this code in published research, please cite:

```bibtex
@software{mirfendereski_dpps_2024,
  author = {Mirfendereski, Siamak},
  title = {Polydisperse Diffusiophoretic Hard-Sphere Simulation},
  year = {2024},
  note = {Python translation by Anthropic Claude}
}
```

---

## License

*[Specify license based on original Fortran code]*

---

## Contact

For questions about:
- **Original Fortran code**: Contact Siamak Mirfendereski
- **Python translation**: Refer to this documentation

---

## References

1. Prigogine, I., & Nicolis, G. (1967). On symmetry-breaking instabilities in dissipative systems. *The Journal of Chemical Physics*, 46(9), 3542-3550.

2. Derjaguin, B. V., et al. (1947). Theory of the stability of strongly charged lyophobic sols. *Acta Physicochim. URSS*, 14, 633-662.

3. Anderson, J. L. (1989). Colloid transport by interfacial forces. *Annual Review of Fluid Mechanics*, 21(1), 61-99.

4. Allen, M. P., & Tildesley, D. J. (2017). *Computer simulation of liquids*. Oxford University Press.

---

## Appendix: Parameter Sensitivity

### Brusselator Parameters
- **A**: Controls base concentration. Typical range: 2-6
- **μ**: Bifurcation parameter. Turing patterns emerge for μ > μ_critical
- **D_rd**: Ratio of diffusion coefficients. Must be > ~2 for patterns

### Mobility Parameters
- Positive mobility: particles move up gradient
- Negative mobility: particles move down gradient
- Opposite signs for C1/C2 lead to interesting segregation

### Péclet Number
- Pe << 1: Diffusion-dominated
- Pe >> 1: Advection-dominated
- Pe ~ 20: Balanced (typical for this system)
