"""
Velocity Calculator
===================

Calculates particle velocities from diffusiophoresis and Brownian motion.
"""

import numpy as np
from typing import Tuple
from scipy.interpolate import RegularGridInterpolator


class VelocityCalculator:
    """
    Computes particle velocities from various physical phenomena:
    - Diffusiophoresis (driven by concentration gradients)
    - Brownian motion (thermal fluctuations)
    """
    
    def __init__(
        self,
        N: int,
        N1: int,
        box: np.ndarray,
        Pe: float,
        eta: float,
        A_rd: float
    ):
        """
        Initialize velocity calculator.
        
        Parameters
        ----------
        N : int
            Total number of particles
        N1 : int
            Number of type 1 particles
        box : np.ndarray
            Box dimensions
        Pe : float
            Peclet number
        eta, A_rd : float
            Reaction-diffusion parameters
        """
        self.N = N
        self.N1 = N1
        self.box = box.copy()
        self.Pe = Pe
        self.eta = eta
        self.A_rd = A_rd
        
        # Random seed for Brownian motion
        self.rng = np.random.default_rng()
        
    def calculate_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        Dx_C1: np.ndarray,
        Dy_C1: np.ndarray,
        Dx_C2: np.ndarray,
        Dy_C2: np.ndarray,
        nx_rd: int,
        ny_rd: int,
        Mob_C1: float,
        Mob_C2: float,
        Mob2_C1: float,
        Mob2_C2: float,
        alpha2: float,
        delt: float,
        a_L0: float,
        size_particle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate total particle velocities.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Particle positions
        Dx_C1, Dy_C1, Dx_C2, Dy_C2 : np.ndarray
            Concentration gradients on grid
        nx_rd, ny_rd : int
            Grid size
        Mob_C1, Mob_C2 : float
            Mobilities for type 1
        Mob2_C1, Mob2_C2 : float
            Mobilities for type 2
        alpha2 : float
            Perturbation amplitude
        delt : float
            Time step
        a_L0 : float
            Scaling factor
        size_particle : np.ndarray
            Particle radii
            
        Returns
        -------
        ux, uy, uz : np.ndarray
            Particle velocities
        """
        # Interpolate concentration gradients to particle positions
        gradx_C1, grady_C1, gradx_C2, grady_C2 = self._interpolate_gradients(
            x, y, Dx_C1, Dy_C1, Dx_C2, Dy_C2, nx_rd, ny_rd
        )
        
        # Diffusiophoretic velocities
        ux_dp, uy_dp, uz_dp = self._diffusiophoresis(
            gradx_C1, grady_C1, gradx_C2, grady_C2,
            Mob_C1, Mob_C2, Mob2_C1, Mob2_C2
        )
        
        # Brownian velocities
        ubrx, ubry, ubrz = self._brownian_motion(delt)
        
        # Brownian normalization factor
        br_normal = np.sqrt(
            alpha2 * (Mob_C1 - Mob_C2 * self.eta * (1 + self.A_rd * self.eta) / self.A_rd) / 
            self.Pe
        )
        
        # Total velocity (size-dependent Brownian motion)
        ux = ux_dp + ubrx * br_normal / np.sqrt(size_particle)
        uy = uy_dp + ubry * br_normal / np.sqrt(size_particle)
        uz = np.zeros(self.N)  # 2D simulation
        
        return ux, uy, uz
    
    def _interpolate_gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Dx_C1: np.ndarray,
        Dy_C1: np.ndarray,
        Dx_C2: np.ndarray,
        Dy_C2: np.ndarray,
        nx_rd: int,
        ny_rd: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate concentration gradients from grid to particle positions.
        Uses bilinear interpolation.
        """
        dx = self.box[0] / nx_rd
        dy = self.box[1] / ny_rd
        
        gradx_C1 = np.zeros(self.N)
        grady_C1 = np.zeros(self.N)
        gradx_C2 = np.zeros(self.N)
        grady_C2 = np.zeros(self.N)
        
        for i in range(self.N):
            # Find cell indices (staggered grid)
            low_x = int(np.floor(x[i] / dx + 1.5))
            x_low_x = (low_x - 1.5) * dx
            high_x = low_x + 1
            x_high_x = (high_x - 1.5) * dx
            
            low_y = int(np.floor(y[i] / dy + 1.5))
            y_low_y = (low_y - 1.5) * dy
            high_y = low_y + 1
            y_high_y = (high_y - 1.5) * dy
            
            # Ensure indices are in bounds (with periodic BC)
            low_x = low_x % (nx_rd + 2)
            high_x = high_x % (nx_rd + 2)
            low_y = low_y % (ny_rd + 2)
            high_y = high_y % (ny_rd + 2)
            
            # Bilinear interpolation weights
            wx_low = (x_high_x - x[i])
            wx_high = (x[i] - x_low_x)
            wy_low = (y_high_y - y[i])
            wy_high = (y[i] - y_low_y)
            
            coef = 1.0 / (dx * dy)
            
            # Interpolate Dx_C1
            first_1 = Dx_C1[low_x, low_y] * wy_low + Dx_C1[low_x, high_y] * wy_high
            first_2 = Dx_C1[high_x, low_y] * wy_low + Dx_C1[high_x, high_y] * wy_high
            gradx_C1[i] = coef * (wx_low * first_1 + wx_high * first_2)
            
            # Interpolate Dy_C1
            first_1 = Dy_C1[low_x, low_y] * wy_low + Dy_C1[low_x, high_y] * wy_high
            first_2 = Dy_C1[high_x, low_y] * wy_low + Dy_C1[high_x, high_y] * wy_high
            grady_C1[i] = coef * (wx_low * first_1 + wx_high * first_2)
            
            # Interpolate Dx_C2
            first_1 = Dx_C2[low_x, low_y] * wy_low + Dx_C2[low_x, high_y] * wy_high
            first_2 = Dx_C2[high_x, low_y] * wy_low + Dx_C2[high_x, high_y] * wy_high
            gradx_C2[i] = coef * (wx_low * first_1 + wx_high * first_2)
            
            # Interpolate Dy_C2
            first_1 = Dy_C2[low_x, low_y] * wy_low + Dy_C2[low_x, high_y] * wy_high
            first_2 = Dy_C2[high_x, low_y] * wy_low + Dy_C2[high_x, high_y] * wy_high
            grady_C2[i] = coef * (wx_low * first_1 + wx_high * first_2)
        
        return gradx_C1, grady_C1, gradx_C2, grady_C2
    
    def _diffusiophoresis(
        self,
        gradx_C1: np.ndarray,
        grady_C1: np.ndarray,
        gradx_C2: np.ndarray,
        grady_C2: np.ndarray,
        Mob_C1: float,
        Mob_C2: float,
        Mob2_C1: float,
        Mob2_C2: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate diffusiophoretic velocities.
        
        V_dp = Mob_C1 * ∇C1 + Mob_C2 * ∇C2
        
        Different mobilities for type 1 and type 2 particles.
        """
        ux_dp = np.zeros(self.N)
        uy_dp = np.zeros(self.N)
        uz_dp = np.zeros(self.N)
        
        # Type 1 particles
        ux_dp[:self.N1] = Mob_C1 * gradx_C1[:self.N1] + Mob_C2 * gradx_C2[:self.N1]
        uy_dp[:self.N1] = Mob_C1 * grady_C1[:self.N1] + Mob_C2 * grady_C2[:self.N1]
        
        # Type 2 particles
        if self.N1 < self.N:
            ux_dp[self.N1:] = Mob2_C1 * gradx_C1[self.N1:] + Mob2_C2 * gradx_C2[self.N1:]
            uy_dp[self.N1:] = Mob2_C1 * grady_C1[self.N1:] + Mob2_C2 * grady_C2[self.N1:]
        
        return ux_dp, uy_dp, uz_dp
    
    def _brownian_motion(self, delt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Brownian velocities.
        
        Brownian forces are Gaussian random variables with zero mean
        and unit variance, scaled by sqrt(2/dt).
        """
        # Generate Gaussian random forces (Box-Muller or built-in normal)
        Wx = self.rng.standard_normal(self.N)
        Wy = self.rng.standard_normal(self.N)
        Wz = self.rng.standard_normal(self.N)
        
        # Brownian velocities (simplified - no hydrodynamic interactions)
        # In original code, there would be mobility matrix multiplication
        # Here we use: u_br = sqrt(2*kT/(m*dt)) * W
        # Simplified to: u_br = sqrt(2/dt) * W (dimensionless units)
        
        scale = np.sqrt(2.0 / delt)
        
        ubrx = scale * Wx
        ubry = scale * Wy
        ubrz = scale * Wz
        
        return ubrx, ubry, ubrz


def gasdev(seed: int = None) -> float:
    """
    Generate Gaussian random deviate with zero mean and unit variance.
    Uses Box-Muller transform.
    
    This is a simplified version - in production, use numpy's built-in.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Box-Muller transform
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    
    return z0
