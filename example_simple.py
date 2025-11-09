#!/usr/bin/env python3
"""
Simple Example: Quick Test of Reaction-Diffusion Particle Simulation
====================================================================

This script demonstrates a minimal working example with reduced parameters
for quick testing (~10-30 minutes runtime).
"""

import numpy as np
from pathlib import Path
from rd_dpps_simulation import PolyDispDPBrownianSimulation
from utils import (
    analyze_results, 
    visualize_particles,
    plot_mean_square_displacement,
    plot_velocity_autocorrelation
)
from reaction_diffusion import visualize_turing_pattern


def quick_test():
    """Run a quick test with minimal parameters."""
    
    print("="*60)
    print("QUICK TEST: Reduced Parameter Simulation")
    print("="*60)
    
    # Create simulation with significantly reduced parameters
    sim = PolyDispDPBrownianSimulation(
        # Reduced particle numbers
        N=2000,              # Instead of 180,000
        N1=1000,             # Instead of 90,000
        
        # Reduced grid resolution
        nx_rd=128,           # Instead of 640
        ny_rd=128,
        
        # Reduced time parameters
        max_step_rd=50000,   # Instead of 6,800,000
        maxc_step=10000,     # Instead of 3,200,000
        t_final=100.0,       # Instead of 2000.0
        delta_t=0.1,         # Slightly larger time step
        
        # Keep same physical parameters
        Pe=20.0,
        A_rd=4.5,
        mu=0.04,
        D_rd=8.0,
        Da_c=1.0,
        
        # Mobility parameters
        Mob_C1=0.1,
        Mob_C2=-0.1,
        Mob2_C1=-0.1,
        Mob2_C2=0.1,
        
        # Output
        output_dir="quick_test_output",
        save_interval=20      # Save less frequently
    )
    
    # Run simulation
    print("\n1. Assigning particle sizes...")
    sim.assign_particle_sizes()
    
    print("\n2. Assigning particle positions...")
    area_fraction = sim.assign_particle_positions()
    
    # Visualize initial configuration
    print("\n3. Visualizing initial configuration...")
    visualize_particles(
        sim.x, sim.y, sim.size_particle, sim.N1, sim.box,
        output_file="quick_test_output/initial_particles.png"
    )
    
    print("\n4. Solving reaction-diffusion equations...")
    sim.solve_reaction_diffusion()
    
    # Visualize Turing pattern
    visualize_turing_pattern(
        sim.C1, sim.C2,
        output_file="quick_test_output/turing_pattern.png"
    )
    
    print("\n5. Solving advection-diffusion equations...")
    sim.solve_advection_diffusion(area_fraction)
    
    print("\n6. Saving field solutions...")
    sim.save_rd_solutions()
    
    print("\n7. Running particle dynamics...")
    sim.run_particle_dynamics()
    
    print("\n8. Saving final data...")
    sim.save_final_data()
    
    # Visualize final configuration
    print("\n9. Visualizing final configuration...")
    visualize_particles(
        sim.x, sim.y, sim.size_particle, sim.N1, sim.box,
        output_file="quick_test_output/final_particles.png"
    )
    
    print("\n10. Analyzing results...")
    analyze_results(Path("quick_test_output"))
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE!")
    print("Check 'quick_test_output' directory for results")
    print("="*60)


def medium_test():
    """Run a medium-scale test (couple hours)."""
    
    print("="*60)
    print("MEDIUM TEST: Intermediate Parameter Simulation")
    print("="*60)
    
    sim = PolyDispDPBrownianSimulation(
        N=20000,
        N1=10000,
        nx_rd=320,
        ny_rd=320,
        max_step_rd=500000,
        maxc_step=100000,
        t_final=500.0,
        output_dir="medium_test_output",
        save_interval=50
    )
    
    sim.run()
    analyze_results(Path("medium_test_output"))
    
    print("\n" + "="*60)
    print("MEDIUM TEST COMPLETE!")
    print("="*60)


def production_run():
    """Run full-scale production simulation (days)."""
    
    print("="*60)
    print("PRODUCTION RUN: Full-Scale Simulation")
    print("WARNING: This will take days to complete!")
    print("="*60)
    
    # Confirm user wants to proceed
    response = input("Are you sure you want to run full simulation? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    sim = PolyDispDPBrownianSimulation(
        # Full parameters from original Fortran
        N=180000,
        N1=90000,
        nx_rd=640,
        ny_rd=640,
        max_step_rd=6800000,
        maxc_step=3200000,
        t_final=2000.0,
        delta_t=0.05,
        output_dir="production_output",
        save_interval=60
    )
    
    sim.run()
    analyze_results(Path("production_output"))
    
    print("\n" + "="*60)
    print("PRODUCTION RUN COMPLETE!")
    print("="*60)


def demo_reaction_diffusion_only():
    """Demonstrate just the reaction-diffusion solver."""
    
    print("="*60)
    print("DEMO: Reaction-Diffusion Solver Only")
    print("="*60)
    
    from reaction_diffusion import ReactionDiffusionSolver
    
    solver = ReactionDiffusionSolver(
        nx=256,
        ny=256,
        size_x=32.0,
        size_y=32.0,
        dt=0.00003,
        max_steps=100000,
        tol=1e-12,
        A=4.5,
        D_rd=8.0,
        mu=0.04,
        Da_c=1.0,
        am_noise=0.02
    )
    
    print("\nSolving Brusselator equations...")
    C1, C2, Dx_C1, Dx_C2, Dy_C1, Dy_C2, alpha = solver.solve()
    
    print(f"\nResults:")
    print(f"  Alpha (perturbation amplitude): {alpha:.6f}")
    print(f"  C1 mean: {np.mean(C1):.6f}")
    print(f"  C1 std:  {np.std(C1):.6f}")
    print(f"  C2 mean: {np.mean(C2):.6f}")
    print(f"  C2 std:  {np.std(C2):.6f}")
    
    visualize_turing_pattern(C1, C2, "rd_demo_pattern.png")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE! See 'rd_demo_pattern.png'")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nAvailable modes:")
        print("  python example_simple.py quick       # Quick test (~15 min)")
        print("  python example_simple.py medium      # Medium test (~2 hours)")
        print("  python example_simple.py production  # Full simulation (days)")
        print("  python example_simple.py rd_only     # Demo RD solver only")
        print()
        mode = input("Select mode (quick/medium/production/rd_only): ").lower()
    
    if mode == "quick":
        quick_test()
    elif mode == "medium":
        medium_test()
    elif mode == "production":
        production_run()
    elif mode == "rd_only":
        demo_reaction_diffusion_only()
    else:
        print(f"Unknown mode: {mode}")
        print("Use: quick, medium, production, or rd_only")
        sys.exit(1)
