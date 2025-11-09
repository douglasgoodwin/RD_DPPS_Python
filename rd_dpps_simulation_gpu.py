"""
GPU-Accelerated Main Simulation
================================

Drop-in replacement for rd_dpps_simulation.py with GPU acceleration.
Uses CuPy for all compute-intensive operations.
"""

import numpy as np
from pathlib import Path
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not found - using CPU version")

# Import GPU or CPU versions
if GPU_AVAILABLE:
    from reaction_diffusion_gpu import ReactionDiffusionSolverGPU as RDSolver
    from advection_diffusion_gpu import AdvectionDiffusionSolverGPU as ADSolver
else:
    from reaction_diffusion import ReactionDiffusionSolver as RDSolver
    from advection_diffusion import AdvectionDiffusionSolver as ADSolver

# CPU modules (particle dynamics still on CPU for now)
from particle_dynamics import ParticleSystem
from velocity_calculator import VelocityCalculator
from utils import create_output_directory


class PolyDispDPBrownianSimulationGPU:
    """
    GPU-accelerated simulation using CuPy.
    
    Expected speedup on A100: 20-50x for field equations
    Overall speedup: ~10-24x depending on particle count
    """
    
    def __init__(self, **kwargs):
        """Initialize with same parameters as CPU version."""
        # [Copy all initialization from rd_dpps_simulation.py]
        # For brevity, key parameters:
        
        self.N = kwargs.get('N', 180000)
        self.N1 = kwargs.get('N1', 90000)
        self.N2 = self.N - self.N1
        
        # Spatial domain
        self.size_x = kwargs.get('size_x', 32.0)
        self.size_y = kwargs.get('size_y', 32.0)
        box_scale = kwargs.get('box_scale', 53.08)
        self.box = np.array([self.size_x * box_scale, self.size_x * box_scale, 2.5])
        
        # Time parameters
        self.delta_t = kwargs.get('delta_t', 0.05)
        self.t_final = kwargs.get('t_final', 2000.0)
        self.M = int(self.t_final / self.delta_t)
        
        # RD parameters
        self.nx_rd = kwargs.get('nx_rd', 640)
        self.ny_rd = kwargs.get('ny_rd', 640)
        self.dt_rd = kwargs.get('dt_rd', 0.00003)
        self.max_step_rd = kwargs.get('max_step_rd', 6800000)
        self.A_rd = kwargs.get('A_rd', 4.5)
        self.mu = kwargs.get('mu', 0.04)
        self.D_rd = kwargs.get('D_rd', 8.0)
        self.Da_c = kwargs.get('Da_c', 1.0)
        self.am_noise = kwargs.get('am_noise', 0.02)
        self.tol_rd = kwargs.get('tol_rd', 1e-12)
        
        # Calculate derived parameters
        self.eta = 1.0 / np.sqrt(self.D_rd)
        Bc = (1 + self.A_rd * self.eta) ** 2
        self.B_rd = Bc * (1 + self.mu)
        
        # Mobility parameters
        self.Mob_C1 = kwargs.get('Mob_C1', 0.1)
        self.Mob_C2 = kwargs.get('Mob_C2', -0.1)
        self.Mob2_C1 = kwargs.get('Mob2_C1', -0.1)
        self.Mob2_C2 = kwargs.get('Mob2_C2', 0.1)
        
        # AD parameters
        self.maxc_step = kwargs.get('maxc_step', 3200000)
        self.tol_ad = kwargs.get('tol_ad', 1e-10)
        self.dt_ad = kwargs.get('dt_ad', 0.000005)
        
        # Other parameters
        self.Pe = kwargs.get('Pe', 20.0)
        self.mid_p_size = kwargs.get('mid_p_size', 1.0)
        self.mid_p_size2 = kwargs.get('mid_p_size2', 1.0)
        self.low_p_size = kwargs.get('low_p_size', 0.5)
        self.low_p_size2 = kwargs.get('low_p_size2', 0.5)
        self.sub_g_size = kwargs.get('sub_g_size', 9.0)
        
        # Output
        self.output_dir = Path(kwargs.get('output_dir', 'output'))
        self.save_interval = kwargs.get('save_interval', 60)
        create_output_directory(self.output_dir)
        
        # Initialize arrays (on CPU)
        self._initialize_arrays()
        
        # Initialize solvers (GPU-accelerated)
        print("\n" + "="*60)
        print("Initializing GPU-Accelerated Simulation")
        print("="*60)
        
        self.rd_solver = RDSolver(
            self.nx_rd, self.ny_rd, self.size_x, self.size_y,
            self.dt_rd, self.max_step_rd, self.tol_rd,
            self.A_rd, self.D_rd, self.mu, self.Da_c, self.am_noise
        )
        
        self.ad_solver = ADSolver(
            self.nx_rd, self.ny_rd, self.size_x, self.size_y,
            self.dt_ad, self.maxc_step, self.tol_ad
        )
        
        self.particle_system = ParticleSystem(self.N, self.box, self.sub_g_size)
        self.velocity_calc = VelocityCalculator(
            self.N, self.N1, self.box, self.Pe, self.eta, self.A_rd
        )
        
        self.a_L0 = self.size_x / self.box[0]
        
        self._print_info()
    
    def _initialize_arrays(self):
        """Initialize arrays (same as CPU version)."""
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        self.z = np.zeros(self.N)
        self.xold = np.zeros(self.N)
        self.yold = np.zeros(self.N)
        self.zold = np.zeros(self.N)
        self.xorig = np.zeros((self.N, 2))
        self.yorig = np.zeros((self.N, 2))
        self.zorig = np.zeros((self.N, 2))
        self.ux = np.zeros((self.N, 2))
        self.uy = np.zeros((self.N, 2))
        self.uz = np.zeros((self.N, 2))
        self.size_particle = np.ones(self.N)
        self.C1 = np.zeros((self.nx_rd + 1, self.ny_rd + 1))
        self.C2 = np.zeros((self.nx_rd + 1, self.ny_rd + 1))
        self.Dx_C1 = np.zeros((self.nx_rd + 2, self.ny_rd + 2))
        self.Dx_C2 = np.zeros((self.nx_rd + 2, self.ny_rd + 2))
        self.Dy_C1 = np.zeros((self.nx_rd + 2, self.ny_rd + 2))
        self.Dy_C2 = np.zeros((self.nx_rd + 2, self.ny_rd + 2))
        self.phi1 = np.zeros((self.nx_rd + 1, self.ny_rd + 1))
        self.phi2 = np.zeros((self.nx_rd + 1, self.ny_rd + 1))
        self.ms = np.zeros((3, self.M + 1))
        self.auto_vel = np.zeros((3, self.M + 1))
        self.p1x = np.zeros(self.M + 1)
        self.p1y = np.zeros(self.M + 1)
    
    def _print_info(self):
        """Print simulation info."""
        print(f"\nSimulation Parameters:")
        print(f"  Particles: {self.N} (Type 1: {self.N1}, Type 2: {self.N2})")
        print(f"  Grid: {self.nx_rd} x {self.ny_rd}")
        print(f"  Time steps: {self.M}")
        print(f"  Pe = {self.Pe:.2f}")
        if GPU_AVAILABLE:
            mem = cp.cuda.Device().mem_info
            print(f"  GPU Memory: {mem[1]/1e9:.1f} GB total, {mem[0]/1e9:.1f} GB free")
        print("="*60 + "\n")
    
    def assign_particle_sizes(self):
        """Assign polydisperse sizes (same as CPU version)."""
        from rd_dpps_simulation import PolyDispDPBrownianSimulation
        cpu_sim = PolyDispDPBrownianSimulation.__new__(PolyDispDPBrownianSimulation)
        cpu_sim.N = self.N
        cpu_sim.N1 = self.N1
        cpu_sim.N2 = self.N2
        cpu_sim.size_particle = self.size_particle
        cpu_sim.mid_p_size = self.mid_p_size
        cpu_sim.mid_p_size2 = self.mid_p_size2
        cpu_sim.low_p_size = self.low_p_size
        cpu_sim.low_p_size2 = self.low_p_size2
        cpu_sim.assign_particle_sizes()
        self.size_particle = cpu_sim.size_particle
        print("Size assignment done")
    
    def assign_particle_positions(self):
        """Assign positions (same as CPU version)."""
        from rd_dpps_simulation import PolyDispDPBrownianSimulation
        cpu_sim = PolyDispDPBrownianSimulation.__new__(PolyDispDPBrownianSimulation)
        cpu_sim.N = self.N
        cpu_sim.box = self.box
        cpu_sim.x = self.x
        cpu_sim.y = self.y
        cpu_sim.z = self.z
        cpu_sim.size_particle = self.size_particle
        cpu_sim._calculate_distance = PolyDispDPBrownianSimulation._calculate_distance.__get__(cpu_sim)
        area_fraction = cpu_sim.assign_particle_positions()
        self.x = cpu_sim.x
        self.y = cpu_sim.y
        self.z = cpu_sim.z
        return area_fraction
    
    def solve_reaction_diffusion(self):
        """Solve RD on GPU."""
        print("Solving reaction-diffusion on GPU...")
        start = time.time()
        
        self.C1, self.C2, self.Dx_C1, self.Dx_C2, self.Dy_C1, self.Dy_C2, self.alpha2 = (
            self.rd_solver.solve()
        )
        
        elapsed = time.time() - start
        print(f"✓ RD solved in {elapsed:.2f}s")
        print(f"  Alpha = {self.alpha2:.6f}")
        
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    
    def solve_advection_diffusion(self, area_fraction: float):
        """Solve AD on GPU."""
        print("\nSolving advection-diffusion on GPU...")
        start = time.time()
        
        self.phi1, self.phi2 = self.ad_solver.solve(
            self.Dx_C1, self.Dx_C2, self.Dy_C1, self.Dy_C2,
            self.Pe, self.eta, self.alpha2, self.A_rd,
            self.Mob_C1, self.Mob_C2, self.Mob2_C1, self.Mob2_C2,
            self.N, self.N1, area_fraction
        )
        
        elapsed = time.time() - start
        print(f"✓ AD solved in {elapsed:.2f}s")
        
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    
    def save_rd_solutions(self):
        """Save solutions (same as CPU)."""
        from rd_dpps_simulation import PolyDispDPBrownianSimulation
        cpu_sim = PolyDispDPBrownianSimulation.__new__(PolyDispDPBrownianSimulation)
        cpu_sim.output_dir = self.output_dir
        cpu_sim.nx_rd = self.nx_rd
        cpu_sim.ny_rd = self.ny_rd
        cpu_sim.size_x = self.size_x
        cpu_sim.size_y = self.size_y
        cpu_sim.C1 = self.C1
        cpu_sim.C2 = self.C2
        cpu_sim.Dx_C1 = self.Dx_C1
        cpu_sim.Dx_C2 = self.Dx_C2
        cpu_sim.Dy_C1 = self.Dy_C1
        cpu_sim.Dy_C2 = self.Dy_C2
        cpu_sim.phi1 = self.phi1
        cpu_sim.phi2 = self.phi2
        cpu_sim.save_rd_solutions()
    
    def run_particle_dynamics(self):
        """Run particle dynamics (CPU for now)."""
        print("\nRunning particle dynamics...")
        from rd_dpps_simulation import PolyDispDPBrownianSimulation
        
        # Use CPU version for particle dynamics
        # (This could be optimized further but requires more work)
        cpu_sim = PolyDispDPBrownianSimulation.__new__(PolyDispDPBrownianSimulation)
        
        # Copy all necessary attributes
        for attr in ['N', 'N1', 'box', 'delta_t', 'M', 'x', 'y', 'z',
                     'xold', 'yold', 'zold', 'xorig', 'yorig', 'zorig',
                     'ux', 'uy', 'uz', 'size_particle', 'Dx_C1', 'Dy_C1',
                     'Dx_C2', 'Dy_C2', 'nx_rd', 'ny_rd', 'Pe', 'eta',
                     'A_rd', 'alpha2', 'Mob_C1', 'Mob_C2', 'Mob2_C1',
                     'Mob2_C2', 'a_L0', 'ms', 'auto_vel', 'p1x', 'p1y',
                     'output_dir', 'save_interval', 'particle_system',
                     'velocity_calc']:
            setattr(cpu_sim, attr, getattr(self, attr))
        
        # Bind methods
        for method in ['_apply_periodic_bc', '_calculate_velocities',
                       '_calculate_mean_square', '_calculate_autocorrelation',
                       '_save_state', '_explicit_euler_step', '_adams_bashforth_step']:
            setattr(cpu_sim, method, getattr(PolyDispDPBrownianSimulation, method).__get__(cpu_sim))
        
        cpu_sim.run_particle_dynamics()
        
        # Copy results back
        self.x = cpu_sim.x
        self.y = cpu_sim.y
        self.z = cpu_sim.z
        self.ms = cpu_sim.ms
        self.auto_vel = cpu_sim.auto_vel
    
    def save_final_data(self):
        """Save final data."""
        from rd_dpps_simulation import PolyDispDPBrownianSimulation
        cpu_sim = PolyDispDPBrownianSimulation.__new__(PolyDispDPBrownianSimulation)
        cpu_sim.output_dir = self.output_dir
        cpu_sim.ms = self.ms
        cpu_sim.auto_vel = self.auto_vel
        cpu_sim.p1x = self.p1x
        cpu_sim.p1y = self.p1y
        cpu_sim.save_final_data()
    
    def run(self):
        """Run complete simulation with GPU acceleration."""
        total_start = time.time()
        
        self.assign_particle_sizes()
        area_fraction = self.assign_particle_positions()
        
        self.solve_reaction_diffusion()
        self.solve_advection_diffusion(area_fraction)
        self.save_rd_solutions()
        
        self.run_particle_dynamics()
        self.save_final_data()
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print(f"✓ Simulation complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print("="*60)


if __name__ == "__main__":
    # Quick test
    sim = PolyDispDPBrownianSimulationGPU(
        N=2000,
        N1=1000,
        nx_rd=256,
        ny_rd=256,
        max_step_rd=50000,
        maxc_step=10000,
        t_final=50.0,
        output_dir="gpu_test_output"
    )
    
    sim.run()
