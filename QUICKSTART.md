# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install required packages
pip install numpy scipy matplotlib numba

# 2. Optional: for animations
pip install ffmpeg-python

# 3. Verify installation
python -c "import numpy, scipy, matplotlib, numba; print('All packages installed!')"
```

## Run Your First Simulation (15 minutes)

```bash
# Quick test with minimal parameters
python example_simple.py quick
```

This will:
1. Create 2,000 particles (instead of 180,000)
2. Use 128×128 grid (instead of 640×640)
3. Run for 100 time units (instead of 2,000)
4. Complete in ~15 minutes
5. Generate plots and data in `quick_test_output/`

## Check Results

After the simulation completes, look in `quick_test_output/` for:

- `turing_pattern.png` - The reaction-diffusion pattern
- `initial_particles.png` - Starting particle configuration
- `final_particles.png` - Ending particle configuration
- `msd.png` - Mean-square displacement (should show diffusive behavior)
- `vacf.png` - Velocity autocorrelation
- `trajectory.png` - Path of particle #1

## Understanding the Output

### Good Results Show:
✓ Turing pattern with spots/stripes
✓ No overlapping particles
✓ MSD increases ~linearly with time
✓ VACF decays to zero
✓ Particles move smoothly

### Troubleshooting:
✗ No pattern → Increase `max_step_rd`
✗ Overlapping particles → Check collision detection
✗ MSD not linear → Check time step size
✗ Simulation too slow → Use smaller N or coarser grid

## Next Steps

### Run Longer Simulation
```bash
python example_simple.py medium  # ~2 hours, 20,000 particles
```

### Customize Parameters
Edit `example_simple.py` and modify:
```python
sim = PolyDispDPBrownianSimulation(
    N=5000,           # Your choice
    Pe=10.0,          # Try different values
    A_rd=5.0,         # Affects pattern type
    # ... etc
)
```

### Just Test Reaction-Diffusion
```bash
python example_simple.py rd_only  # ~2 minutes
```

## Production Run (Days)

```bash
python example_simple.py production
```

Full scale: 180,000 particles, 640×640 grid (same as Fortran original)

## Need Help?

1. Read `README.md` for full documentation
2. Check `TRANSLATION_SUMMARY.md` for implementation details
3. Look at inline comments in source code
4. Compare with original Fortran files

## Files Overview

| File | Purpose |
|------|---------|
| `rd_dpps_simulation.py` | Main simulation class |
| `reaction_diffusion.py` | Brusselator solver |
| `advection_diffusion.py` | Continuum solver |
| `particle_dynamics.py` | Collision detection |
| `velocity_calculator.py` | Forces and velocities |
| `utils.py` | Plotting and analysis |
| `example_simple.py` | Example scripts |
| `README.md` | Full documentation |

## Common Workflows

### Test a Parameter
```python
from rd_dpps_simulation import PolyDispDPBrownianSimulation

# Vary Peclet number
for Pe in [10, 20, 30]:
    sim = PolyDispDPBrownianSimulation(
        N=1000, Pe=Pe, t_final=50,
        output_dir=f"results_Pe{Pe}"
    )
    sim.run()
```

### Analyze Existing Results
```python
from utils import analyze_results
from pathlib import Path

analyze_results(Path("quick_test_output"))
```

### Create Animation
```python
from utils import create_animation
import numpy as np

create_animation(
    Path("quick_test_output"),
    box=np.array([1700, 1700, 2.5]),
    N1=1000,
    size_particle=np.ones(2000),
    output_file="movie.mp4"
)
```

## Performance Tips

### Speed Up:
- Reduce `N` (number of particles)
- Reduce grid resolution (`nx_rd`, `ny_rd`)
- Increase time steps (carefully!)
- Use smaller `t_final` for testing

### Save Memory:
- Increase `save_interval` (save less frequently)
- Reduce `max_p_sub` if possible
- Delete old checkpoint files

### GPU Acceleration (Advanced):
```python
# Install CuPy
pip install cupy-cuda11x  # or cuda12x

# Modify code to use cupy instead of numpy
import cupy as np  # Drop-in replacement for many operations
```

## Validation

Compare with Fortran output (if available):

1. **Turing Pattern:** Should have similar wavelength and amplitude
2. **Particle Distribution:** Similar density patterns
3. **MSD Slope:** Should match diffusion coefficient
4. **Alpha:** Perturbation amplitude should be close

## System Requirements

### Minimum:
- 8 GB RAM
- 2 CPU cores
- Python 3.8+
- ~500 MB disk space

### Recommended:
- 32 GB RAM
- 8+ CPU cores
- Python 3.10+
- ~10 GB disk space (for large runs)

### For Production:
- 64+ GB RAM
- 16+ CPU cores
- Fast SSD
- ~100 GB disk space

## Estimated Runtimes

| Mode | Particles | Grid | Time Steps | Duration |
|------|-----------|------|------------|----------|
| Quick | 2,000 | 128² | 1,000 | 15 min |
| Medium | 20,000 | 320² | 10,000 | 2 hours |
| Production | 180,000 | 640² | 40,000 | 2-7 days |

*Times on modern workstation (16 cores, 32 GB RAM)*

## What's Different from Fortran?

### Advantages:
✓ Easier to read and modify
✓ Built-in visualization
✓ Better error messages
✓ Modular design
✓ Type hints
✓ Comprehensive documentation

### Disadvantages:
✗ ~10-50x slower (varies by section)
✗ Uses more memory
✗ Requires Python ecosystem

### Same Physics:
= Identical algorithms
= Same numerical schemes
= Same boundary conditions
= Same physical parameters
= Results should match within numerical tolerance

## Getting Results Fast

For quick exploration:
```python
# Minimal viable test
sim = PolyDispDPBrownianSimulation(
    N=500, N1=250,
    nx_rd=64, ny_rd=64,
    max_step_rd=10000,
    maxc_step=2000,
    t_final=20.0,
    output_dir="ultra_quick"
)
sim.run()
```

This completes in ~5 minutes and still shows the key physics!

## Success!

If you see:
```
Simulation complete!
```

And output files in your directory, you're ready to go!

Start exploring the parameter space and understanding colloid dynamics.

---

**Questions?** Check README.md or TRANSLATION_SUMMARY.md
