"""
Utilities
=========

Helper functions for file I/O, data management, and visualization.
"""

import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json


def create_output_directory(output_dir: Path):
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")


def save_data(filename: str, data: np.ndarray, header: str = ""):
    """Save numpy array to text file."""
    np.savetxt(filename, data, header=header, fmt='%.10e')


def load_checkpoint(
    checkpoint_dir: Path,
    N: int
) -> tuple:
    """
    Load particle positions and velocities from checkpoint files.
    
    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoint files
    N : int
        Number of particles
        
    Returns
    -------
    x, y, z, ux, uy, uz : np.ndarray
        Particle positions and velocities
    file_count : int
        Number of checkpoint file loaded
    """
    # Find the most recent checkpoint
    pos_files = list(checkpoint_dir.glob("pos*.txt"))
    
    if not pos_files:
        return None, None, None, None, None, None, 0
    
    # Get the highest numbered file
    file_nums = [int(f.stem[3:]) for f in pos_files]
    max_num = max(file_nums)
    
    # Load positions and velocities
    pos_file = checkpoint_dir / f"pos{max_num:04d}.txt"
    vel_file = checkpoint_dir / f"vel{max_num:04d}.txt"
    
    if pos_file.exists() and vel_file.exists():
        pos_data = np.loadtxt(pos_file)
        vel_data = np.loadtxt(vel_file)
        
        x = pos_data[:N, 0]
        y = pos_data[:N, 1]
        z = pos_data[:N, 2]
        ux = vel_data[:N, 0]
        uy = vel_data[:N, 1]
        uz = vel_data[:N, 2]
        
        print(f"Loaded checkpoint from file {max_num}")
        return x, y, z, ux, uy, uz, max_num
    
    return None, None, None, None, None, None, 0


def save_simulation_params(output_dir: Path, params: Dict[str, Any]):
    """Save simulation parameters to JSON file."""
    param_file = output_dir / "simulation_params.json"
    
    # Convert numpy types to native Python types
    clean_params = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            clean_params[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            clean_params[key] = value.item()
        else:
            clean_params[key] = value
    
    with open(param_file, 'w') as f:
        json.dump(clean_params, f, indent=2)
    
    print(f"Saved parameters to {param_file}")


def load_simulation_params(param_file: Path) -> Dict[str, Any]:
    """Load simulation parameters from JSON file."""
    with open(param_file, 'r') as f:
        params = json.load(f)
    return params


def visualize_particles(
    x: np.ndarray,
    y: np.ndarray,
    size_particle: np.ndarray,
    N1: int,
    box: np.ndarray,
    output_file: str = "particles.png",
    show_box: bool = True
):
    """
    Visualize particle positions.
    
    Parameters
    ----------
    x, y : np.ndarray
        Particle positions
    size_particle : np.ndarray
        Particle radii
    N1 : int
        Number of type 1 particles (different color)
    box : np.ndarray
        Box dimensions
    output_file : str
        Output filename
    show_box : bool
        Whether to show box boundaries
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot type 1 particles
    ax.scatter(
        x[:N1], y[:N1], 
        s=size_particle[:N1]**2 * 100, 
        c='blue', 
        alpha=0.6, 
        label='Type 1'
    )
    
    # Plot type 2 particles
    if N1 < len(x):
        ax.scatter(
            x[N1:], y[N1:], 
            s=size_particle[N1:]**2 * 100, 
            c='red', 
            alpha=0.6, 
            label='Type 2'
        )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend()
    
    if show_box:
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=box[1], color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=box[0], color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"Saved particle visualization to {output_file}")


def plot_mean_square_displacement(
    ms: np.ndarray,
    dt: float,
    output_file: str = "msd.png"
):
    """
    Plot mean-square displacement vs time.
    
    Parameters
    ----------
    ms : np.ndarray
        Mean-square displacements (3 x M)
    dt : float
        Time step
    output_file : str
        Output filename
    """
    M = ms.shape[1]
    t = np.arange(M) * dt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, ms[0, :], label='x-direction', linewidth=2)
    ax.plot(t, ms[1, :], label='y-direction', linewidth=2)
    ax.plot(t, ms[2, :], label='z-direction', linewidth=2)
    ax.plot(t, ms[0, :] + ms[1, :] + ms[2, :], 
            label='Total', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mean Square Displacement', fontsize=12)
    ax.set_title('Mean Square Displacement vs Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-log plot for diffusion analysis
    ax_inset = fig.add_axes([0.55, 0.2, 0.3, 0.3])
    total_msd = ms[0, :] + ms[1, :] + ms[2, :]
    valid = (t > 0) & (total_msd > 0)
    ax_inset.loglog(t[valid], total_msd[valid], 'k-', linewidth=1.5)
    ax_inset.set_xlabel('Time (log)', fontsize=9)
    ax_inset.set_ylabel('MSD (log)', fontsize=9)
    ax_inset.grid(True, alpha=0.3, which='both')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved MSD plot to {output_file}")


def plot_velocity_autocorrelation(
    auto_vel: np.ndarray,
    dt: float,
    output_file: str = "vacf.png"
):
    """
    Plot velocity autocorrelation function.
    
    Parameters
    ----------
    auto_vel : np.ndarray
        Velocity autocorrelations (3 x M)
    dt : float
        Time step
    output_file : str
        Output filename
    """
    M = auto_vel.shape[1]
    t = np.arange(M) * dt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize by initial value
    ax.plot(t, auto_vel[0, :] / auto_vel[0, 0], label='x-direction', linewidth=2)
    ax.plot(t, auto_vel[1, :] / auto_vel[1, 0], label='y-direction', linewidth=2)
    ax.plot(t, auto_vel[2, :] / auto_vel[2, 0], label='z-direction', linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized VACF', fontsize=12)
    ax.set_title('Velocity Autocorrelation Function', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved VACF plot to {output_file}")


def plot_particle_trajectory(
    p1x: np.ndarray,
    p1y: np.ndarray,
    box: np.ndarray,
    output_file: str = "trajectory.png"
):
    """
    Plot trajectory of a single particle.
    
    Parameters
    ----------
    p1x, p1y : np.ndarray
        Particle trajectory coordinates
    box : np.ndarray
        Box dimensions
    output_file : str
        Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectory
    ax.plot(p1x, p1y, 'b-', alpha=0.6, linewidth=1)
    ax.plot(p1x[0], p1y[0], 'go', markersize=10, label='Start')
    ax.plot(p1x[-1], p1y[-1], 'ro', markersize=10, label='End')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Particle Trajectory', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory plot to {output_file}")


def create_animation(
    output_dir: Path,
    box: np.ndarray,
    N1: int,
    size_particle: np.ndarray,
    output_file: str = "animation.mp4",
    max_frames: int = 100
):
    """
    Create animation of particle motion from saved snapshots.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing position files
    box : np.ndarray
        Box dimensions
    N1 : int
        Number of type 1 particles
    size_particle : np.ndarray
        Particle radii
    output_file : str
        Output filename
    max_frames : int
        Maximum number of frames to include
    """
    # Find all position files
    pos_files = sorted(output_dir.glob("pos*.txt"))
    
    if not pos_files:
        print("No position files found for animation")
        return
    
    # Subsample if too many files
    if len(pos_files) > max_frames:
        step = len(pos_files) // max_frames
        pos_files = pos_files[::step]
    
    print(f"Creating animation with {len(pos_files)} frames...")
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        
        # Load positions
        pos_data = np.loadtxt(pos_files[frame])
        x = pos_data[:, 0]
        y = pos_data[:, 1]
        
        # Plot particles
        ax.scatter(
            x[:N1], y[:N1],
            s=size_particle[:N1]**2 * 100,
            c='blue', alpha=0.6, label='Type 1'
        )
        
        if N1 < len(x):
            ax.scatter(
                x[N1:], y[N1:],
                s=size_particle[N1:]**2 * 100,
                c='red', alpha=0.6, label='Type 2'
            )
        
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Frame {frame+1}/{len(pos_files)}')
        ax.legend()
        
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(pos_files), 
                        interval=100, blit=False)
    
    # Save animation
    anim.save(output_dir / output_file, writer='ffmpeg', fps=10, dpi=100)
    plt.close()
    
    print(f"Saved animation to {output_dir / output_file}")


def analyze_results(output_dir: Path):
    """
    Perform comprehensive analysis of simulation results.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing simulation output
    """
    print(f"\nAnalyzing results from {output_dir}...")
    
    # Load mean-square displacements
    ms_file = output_dir / "ms.txt"
    if ms_file.exists():
        ms = np.loadtxt(ms_file).T
        plot_mean_square_displacement(ms, 0.05, output_dir / "msd.png")
    
    # Load velocity autocorrelations
    auto_file = output_dir / "autocorrelation_vel.txt"
    if auto_file.exists():
        auto_vel = np.loadtxt(auto_file).T
        plot_velocity_autocorrelation(auto_vel, 0.05, output_dir / "vacf.png")
    
    # Load trajectory
    traj_file = output_dir / "par1.txt"
    if traj_file.exists():
        traj = np.loadtxt(traj_file)
        # Assuming box size (need to load from params)
        box = np.array([1700, 1700, 2.5])
        plot_particle_trajectory(traj[:, 0], traj[:, 1], box, 
                                output_dir / "trajectory.png")
    
    print("Analysis complete!")
