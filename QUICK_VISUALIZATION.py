"""
Quick Visualization Script
===========================
Run this single cell in Colab to visualize your simulation!
"""

# 🚀 QUICK VISUALIZATION - ONE CELL SOLUTION
# ===========================================
# Just run this cell and get all visualizations!

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, display

# 1. Set your output directory
output_dir = Path('/content/gpu_output')  # Change if needed

print("🎨 Creating visualizations...")
print(f"Looking in: {output_dir}\n")

# 2. Load basic info
pos_files = sorted(output_dir.glob('pos*.txt'))
print(f"✓ Found {len(pos_files)} snapshots")

# Load first position file
positions = np.loadtxt(pos_files[0])
N = len(positions)
print(f"✓ {N} particles")

# Estimate box size
box_x = positions[:, 0].max() * 1.1
box_y = positions[:, 1].max() * 1.1

# Load particle radii (or default to 1.0)
radius_file = output_dir / 'particle_radius.txt'
radii = np.loadtxt(radius_file) if radius_file.exists() else np.ones(N)

# 3. CREATE ALL VISUALIZATIONS
# ==============================

# --- Particle Snapshots ---
print("\n📸 Creating particle snapshots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, ax in zip([0, len(pos_files)//2, -1], axes):
    pos = np.loadtxt(pos_files[idx])
    N1 = N // 2
    ax.scatter(pos[:N1, 0], pos[:N1, 1], c='blue', s=20, alpha=0.6, label='Type 1')
    ax.scatter(pos[N1:, 0], pos[N1:, 1], c='red', s=20, alpha=0.6, label='Type 2')
    ax.set_xlim(0, box_x)
    ax.set_ylim(0, box_y)
    ax.set_aspect('equal')
    ax.set_title(f'Snapshot {idx if idx >= 0 else len(pos_files)+idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'snapshots.png', dpi=150)
plt.show()
print("✓ Saved: snapshots.png")

# --- Turing Pattern ---
print("\n🌈 Plotting Turing pattern...")
rd_file = output_dir / 'Brusselator_SS.txt'
if rd_file.exists():
    data = np.loadtxt(rd_file)
    nx = len(np.unique(data[:, 0]))
    ny = len(np.unique(data[:, 1]))
    C1 = data[:, 2].reshape(nx, ny)
    C2 = data[:, 3].reshape(nx, ny)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(C1.T, origin='lower', cmap='viridis')
    axes[0].set_title('Species C1')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(C2.T, origin='lower', cmap='plasma')
    axes[1].set_title('Species C2')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'turing_pattern.png', dpi=150)
    plt.show()
    print("✓ Saved: turing_pattern.png")

# --- Mean-Square Displacement ---
print("\n📈 Plotting MSD...")
ms_file = output_dir / 'ms.txt'
if ms_file.exists():
    ms = np.loadtxt(ms_file)
    t = np.arange(len(ms)) * 0.05
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, ms[:, 0] + ms[:, 1], 'k-', linewidth=2, label='Total MSD')
    plt.xlabel('Time')
    plt.ylabel('Mean Square Displacement')
    plt.title('Mean Square Displacement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'msd.png', dpi=150)
    plt.show()
    print("✓ Saved: msd.png")

# --- Velocity Autocorrelation ---
print("\n📉 Plotting VACF...")
vacf_file = output_dir / 'autocorrelation_vel.txt'
if vacf_file.exists():
    vacf = np.loadtxt(vacf_file)
    t = np.arange(len(vacf)) * 0.05
    
    plt.figure(figsize=(10, 6))
    if vacf[0, 0] != 0:
        plt.plot(t, vacf[:, 0] / vacf[0, 0], linewidth=2, label='x')
    if vacf[0, 1] != 0:
        plt.plot(t, vacf[:, 1] / vacf[0, 1], linewidth=2, label='y')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Normalized VACF')
    plt.title('Velocity Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'vacf.png', dpi=150)
    plt.show()
    print("✓ Saved: vacf.png")

# --- Particle Trajectory ---
print("\n🛤️  Plotting trajectory...")
traj_file = output_dir / 'par1.txt'
if traj_file.exists():
    traj = np.loadtxt(traj_file)
    
    plt.figure(figsize=(10, 10))
    plt.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.6)
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
    plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=12, label='End')
    plt.xlim(0, box_x)
    plt.ylim(0, box_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle 1 Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.savefig(output_dir / 'trajectory.png', dpi=150)
    plt.show()
    print("✓ Saved: trajectory.png")

# 4. DISPLAY ALL IMAGES
print("\n" + "="*60)
print("📊 ALL VISUALIZATIONS")
print("="*60)

for img in sorted(output_dir.glob('*.png')):
    print(f"\n{img.name}:")
    display(Image(filename=str(img), width=900))

print("\n✅ Done! All visualizations created and displayed.")
print(f"📁 Files saved in: {output_dir}")
