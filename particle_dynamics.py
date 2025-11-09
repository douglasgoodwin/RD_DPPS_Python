"""
Particle Dynamics System
=========================

Handles particle collisions, subcell grid management, and hard-sphere interactions.
"""

import numpy as np
from typing import Tuple
from numba import jit, prange
import warnings


class ParticleSystem:
    """
    Manages particle positions and resolves hard-sphere collisions
    using a subcell grid algorithm for efficient contact detection.
    """
    
    def __init__(self, N: int, box: np.ndarray, sub_g_size: float):
        """
        Initialize particle system.
        
        Parameters
        ----------
        N : int
            Number of particles
        box : np.ndarray
            Box dimensions [Lx, Ly, Lz]
        sub_g_size : float
            Size of subcells for collision detection
        """
        self.N = N
        self.box = box.copy()
        
        # Subcell grid parameters
        self.nx_sub = int(box[0] / sub_g_size)
        self.ny_sub = int(box[1] / sub_g_size)
        self.dx_sub = box[0] / self.nx_sub
        self.dy_sub = box[1] / self.ny_sub
        
        # Estimate maximum particles per subcell
        self.max_p_sub = int(0.9 * (self.dx_sub * self.dy_sub / np.pi)) + 28
        
        # Subcell grid: stores particle indices in each cell
        self.grid_sub = np.zeros(
            (self.nx_sub, self.ny_sub, self.max_p_sub), 
            dtype=np.int32
        )
        
        # Stack for tracking particles that need collision check
        self.stack = np.zeros(N, dtype=np.int32)
        
        print(f"Subcell grid: {self.nx_sub} x {self.ny_sub}, "
              f"max particles per cell: {self.max_p_sub}")
        
    def initialize(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray,
        size_particle: np.ndarray
    ):
        """Initialize the subcell grid with current particle positions."""
        self._build_grid(x, y, size_particle)
        
    def resolve_collisions(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        size_particle: np.ndarray,
        distance_cutoff: float = 2.005,
        ext_g: float = 0.0025,
        max_iter: int = 100
    ):
        """
        Detect and resolve particle collisions using fast contact algorithm.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Particle positions
        size_particle : np.ndarray
            Particle radii
        distance_cutoff : float
            Cutoff for overlap detection
        ext_g : float
            Extension for subcell search
        max_iter : int
            Maximum iterations per particle
        """
        # Build subcell grid
        self._build_grid(x, y, size_particle, ext_g)
        
        # Process each particle in stack
        is_particle = 0
        
        while True:
            # Find next particle that needs checking
            is_particle = 0
            for i in range(self.N):
                if self.stack[i] == 1:
                    is_particle = i
                    break
            
            if is_particle == 0 and np.all(self.stack == 0):
                break  # All particles processed
            
            overlap_exists = True
            iter_num = 0
            
            # Resolve collisions for this particle
            while overlap_exists and iter_num < max_iter:
                iter_num += 1
                overlap_exists = False
                
                # Get neighboring cells
                x_low = int(np.floor((x[is_particle] - size_particle[is_particle] - ext_g) / self.dx_sub))
                x_high = int(np.floor((x[is_particle] + size_particle[is_particle] + ext_g) / self.dx_sub))
                y_low = int(np.floor((y[is_particle] - size_particle[is_particle] - ext_g) / self.dy_sub))
                y_high = int(np.floor((y[is_particle] + size_particle[is_particle] + ext_g) / self.dy_sub))
                
                # Store old position
                x_old = x[is_particle]
                y_old = y[is_particle]
                
                # Check all neighboring cells
                for i in range(x_low, x_high + 1):
                    for j in range(y_low, y_high + 1):
                        # Apply periodic boundaries
                        i1 = i % self.nx_sub
                        j1 = j % self.ny_sub
                        
                        # Check particles in this cell
                        for k in range(self.max_p_sub):
                            index_j = self.grid_sub[i1, j1, k]
                            
                            if index_j == 0 or index_j == is_particle + 1:
                                continue
                            
                            index_j -= 1  # Convert to 0-indexed
                            
                            # Calculate distance with minimum image
                            xij = x[is_particle] - x[index_j]
                            yij = y[is_particle] - y[index_j]
                            zij = z[is_particle] - z[index_j]
                            
                            xij = xij - self.box[0] * np.round(xij / self.box[0])
                            yij = yij - self.box[1] * np.round(yij / self.box[1])
                            zij = zij - self.box[2] * np.round(zij / self.box[2])
                            
                            Rij = np.sqrt(xij**2 + yij**2 + zij**2)
                            
                            correct_dist_cut = (
                                size_particle[is_particle] + 
                                size_particle[index_j] + 
                                distance_cutoff - 2.0
                            )
                            
                            # If overlapping, push particles apart
                            if Rij < correct_dist_cut:
                                overlap_exists = True
                                
                                # Push particle away from overlap
                                if Rij > 1e-10:
                                    eij_x = xij / Rij
                                    eij_y = yij / Rij
                                    eij_z = zij / Rij
                                else:
                                    # Random direction if exactly on top
                                    angle = np.random.uniform(0, 2*np.pi)
                                    eij_x = np.cos(angle)
                                    eij_y = np.sin(angle)
                                    eij_z = 0.0
                                
                                # Move particle
                                diff = 0.5 * (correct_dist_cut - Rij)
                                x[is_particle] += diff * eij_x
                                y[is_particle] += diff * eij_y
                                z[is_particle] += diff * eij_z
                
                # Update grid if particle moved
                if overlap_exists:
                    self._update_single_particle_grid(
                        is_particle, x, y, size_particle, 
                        x_old, y_old, ext_g
                    )
                
                # Check again if still overlapping
                if overlap_exists:
                    overlap_exists = self._check_overlaps(
                        is_particle, x, y, z, size_particle,
                        distance_cutoff, ext_g
                    )
            
            if iter_num >= max_iter:
                warnings.warn(
                    f"Particle {is_particle} still has overlaps after "
                    f"{max_iter} iterations"
                )
                # Mark all particles for recheck
                self.stack[:] = 1
            
            # Mark this particle as processed
            self.stack[is_particle] = 0
    
    def _build_grid(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        size_particle: np.ndarray,
        ext_g: float = 0.0025
    ):
        """Build the subcell grid from particle positions."""
        self.grid_sub[:, :, :] = 0
        self.stack[:] = 1
        
        count = np.zeros((self.nx_sub, self.ny_sub), dtype=np.int32)
        
        for i in range(self.N):
            # Get cell range this particle occupies
            x_low = int(np.floor((x[i] - size_particle[i] - ext_g) / self.dx_sub))
            x_high = int(np.floor((x[i] + size_particle[i] + ext_g) / self.dx_sub))
            y_low = int(np.floor((y[i] - size_particle[i] - ext_g) / self.dy_sub))
            y_high = int(np.floor((y[i] + size_particle[i] + ext_g) / self.dy_sub))
            
            # Apply periodic boundaries
            x_low = x_low % self.nx_sub
            x_high = x_high % self.nx_sub
            y_low = y_low % self.ny_sub
            y_high = y_high % self.ny_sub
            
            # Add particle to cells
            cells = []
            if x_low == x_high and y_low == y_high:
                cells = [(x_low, y_high)]
            elif x_low == x_high:
                cells = [(x_low, y_low), (x_low, y_high)]
            elif y_low == y_high:
                cells = [(x_low, y_high), (x_high, y_high)]
            else:
                cells = [
                    (x_low, y_low), (x_high, y_low),
                    (x_low, y_high), (x_high, y_high)
                ]
            
            for cx, cy in cells:
                if count[cx, cy] < self.max_p_sub:
                    self.grid_sub[cx, cy, count[cx, cy]] = i + 1  # 1-indexed
                    count[cx, cy] += 1
                else:
                    warnings.warn(
                        f"Subcell ({cx}, {cy}) exceeded max particles "
                        f"({self.max_p_sub})"
                    )
    
    def _update_single_particle_grid(
        self,
        p_index: int,
        x: np.ndarray,
        y: np.ndarray,
        size_particle: np.ndarray,
        x_old: float,
        y_old: float,
        ext_g: float
    ):
        """Update grid after moving a single particle."""
        # Get old cell range
        x_low_old = int(np.floor((x_old - size_particle[p_index] - ext_g) / self.dx_sub))
        x_high_old = int(np.floor((x_old + size_particle[p_index] + ext_g) / self.dx_sub))
        y_low_old = int(np.floor((y_old - size_particle[p_index] - ext_g) / self.dy_sub))
        y_high_old = int(np.floor((y_old + size_particle[p_index] + ext_g) / self.dy_sub))
        
        # Get new cell range
        x_low = int(np.floor((x[p_index] - size_particle[p_index] - ext_g) / self.dx_sub))
        x_high = int(np.floor((x[p_index] + size_particle[p_index] + ext_g) / self.dx_sub))
        y_low = int(np.floor((y[p_index] - size_particle[p_index] - ext_g) / self.dy_sub))
        y_high = int(np.floor((y[p_index] + size_particle[p_index] + ext_g) / self.dy_sub))
        
        # Only update if cells changed
        if (x_low != x_low_old or x_high != x_high_old or 
            y_low != y_low_old or y_high != y_high_old):
            
            # Remove from old cells
            for i in range(x_low_old, x_high_old + 1):
                for j in range(y_low_old, y_high_old + 1):
                    i1 = i % self.nx_sub
                    j1 = j % self.ny_sub
                    
                    for k in range(self.max_p_sub):
                        if self.grid_sub[i1, j1, k] == p_index + 1:
                            self.grid_sub[i1, j1, k] = 0
                            break
            
            # Add to new cells
            for i in range(x_low, x_high + 1):
                for j in range(y_low, y_high + 1):
                    i1 = i % self.nx_sub
                    j1 = j % self.ny_sub
                    
                    # Check if already in this cell
                    already_present = False
                    for k in range(self.max_p_sub):
                        if self.grid_sub[i1, j1, k] == p_index + 1:
                            already_present = True
                            break
                    
                    if not already_present:
                        # Find empty slot
                        for k in range(self.max_p_sub):
                            if self.grid_sub[i1, j1, k] == 0:
                                self.grid_sub[i1, j1, k] = p_index + 1
                                break
    
    def _check_overlaps(
        self,
        is_particle: int,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        size_particle: np.ndarray,
        distance_cutoff: float,
        ext_g: float
    ) -> bool:
        """Check if particle still has overlaps."""
        x_low = int(np.floor((x[is_particle] - size_particle[is_particle] - ext_g) / self.dx_sub))
        x_high = int(np.floor((x[is_particle] + size_particle[is_particle] + ext_g) / self.dx_sub))
        y_low = int(np.floor((y[is_particle] - size_particle[is_particle] - ext_g) / self.dy_sub))
        y_high = int(np.floor((y[is_particle] + size_particle[is_particle] + ext_g) / self.dy_sub))
        
        for i in range(x_low, x_high + 1):
            for j in range(y_low, y_high + 1):
                i1 = i % self.nx_sub
                j1 = j % self.ny_sub
                
                for k in range(self.max_p_sub):
                    index_j = self.grid_sub[i1, j1, k]
                    
                    if index_j == 0 or index_j == is_particle + 1:
                        continue
                    
                    index_j -= 1
                    
                    xij = x[is_particle] - x[index_j]
                    yij = y[is_particle] - y[index_j]
                    zij = z[is_particle] - z[index_j]
                    
                    xij = xij - self.box[0] * np.round(xij / self.box[0])
                    yij = yij - self.box[1] * np.round(yij / self.box[1])
                    zij = zij - self.box[2] * np.round(zij / self.box[2])
                    
                    Rij = np.sqrt(xij**2 + yij**2 + zij**2)
                    
                    correct_dist_cut = (
                        size_particle[is_particle] + 
                        size_particle[index_j] + 
                        distance_cutoff - 2.0
                    )
                    
                    if Rij < correct_dist_cut:
                        return True
        
        return False
