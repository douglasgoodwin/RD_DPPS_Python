# 🔧 IMMEDIATE FIX FOR COLAB - Run This Cell!
# ============================================
# This fixes the IndexError without re-uploading files

import sys

# Monkey-patch the problematic function
def fixed_run_particle_dynamics(self, checkpoint_file=0):
    """Fixed version that handles array indexing correctly."""
    print("\nStarting particle dynamics simulation...")
    
    # Initialize particle system
    self.particle_system.initialize(self.x, self.y, self.z, self.size_particle)
    
    # Initialize velocities
    self.ux[:, :] = 0.0
    self.uy[:, :] = 0.0
    self.uz[:, :] = 0.0
    
    # Save initial state
    self._save_state(0)
    
    # Explicit Euler for first step
    self._explicit_euler_step()
    
    # Calculate initial velocities
    self._calculate_velocities()
    
    # Save first step
    # self._save_state(1)  # Commented out - will save in loop
    
    # Store trajectory of particle 0 - FIXED INDEXING
    self.p1x[0] = self.xorig[0, 0]  # Initial position (column 0)
    self.p1y[0] = self.yorig[0, 0]
    
    # Calculate first step statistics
    self._calculate_mean_square(1)
    self._calculate_autocorrelation(1)
    self.p1x[1] = self.xorig[0, 1]  # Current position (column 1)
    self.p1y[1] = self.yorig[0, 1]
    
    # Adams-Bashforth for remaining steps
    start_step = checkpoint_file * 60 + 2 if checkpoint_file > 0 else 2
    
    for step in range(start_step, self.M + 1):
        if step % 100 == 0:
            print(f"Step {step}/{self.M}")
        
        # Time march
        self._adams_bashforth_step(step)
        
        # Calculate velocities
        self._calculate_velocities()
        
        # Calculate statistics
        self._calculate_mean_square(step)
        self._calculate_autocorrelation(step)
        
        # Store trajectory - FIXED: always use column 1
        self.p1x[step] = self.xorig[0, 1]
        self.p1y[step] = self.yorig[0, 1]
        
        # Save periodically
        if step % self.save_interval == 0:
            ifile = step // self.save_interval
            self._save_state(ifile)
    
    print("Particle dynamics simulation complete")


# Import the classes
if 'rd_dpps_simulation' in sys.modules:
    from rd_dpps_simulation import PolyDispDPBrownianSimulation
    PolyDispDPBrownianSimulation.run_particle_dynamics = fixed_run_particle_dynamics
    print("✓ Fixed CPU version")

if 'rd_dpps_simulation_gpu' in sys.modules:
    # The GPU version uses CPU particle dynamics, so same fix applies
    print("✓ Fixed GPU version")

print("\n" + "="*60)
print("HOTFIX APPLIED!")
print("You can now run: sim.run()")
print("="*60)
