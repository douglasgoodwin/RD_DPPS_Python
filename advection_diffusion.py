"""
Advection-Diffusion Solver
===========================

Finite Difference Method solver for the continuum description 
of colloidal concentration.
"""

import numpy as np
from typing import Tuple


class AdvectionDiffusionSolver:
    """
    FDM solver for advection-diffusion equations describing colloidal transport.
    
    Solves:
        ∂φ/∂t + ∇·(V_dp * φ) = D_eff * ∇²φ
    
    where V_dp is the diffusiophoretic velocity and D_eff is the effective
    diffusion coefficient.
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        size_x: float,
        size_y: float,
        dt: float,
        max_steps: int,
        tol: float
    ):
        """
        Initialize the advection-diffusion solver.
        
        Parameters
        ----------
        nx, ny : int
            Grid points
        size_x, size_y : float
            Domain size
        dt : float
            Time step
        max_steps : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.nx = nx
        self.ny = ny
        self.size_x = size_x
        self.size_y = size_y
        self.dt = dt
        self.max_steps = max_steps
        self.tol = tol
        
        # Grid spacing
        self.dx = size_x / nx
        self.dy = size_y / ny
        
    def solve(
        self,
        Dx_C1: np.ndarray,
        Dx_C2: np.ndarray,
        Dy_C1: np.ndarray,
        Dy_C2: np.ndarray,
        Pe: float,
        eta: float,
        alpha2: float,
        A_rd: float,
        Mob_C1: float,
        Mob_C2: float,
        Mob2_C1: float,
        Mob2_C2: float,
        N: int,
        N1: int,
        area_fraction: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve advection-diffusion equations for colloidal concentrations.
        
        Parameters
        ----------
        Dx_C1, Dx_C2, Dy_C1, Dy_C2 : np.ndarray
            Gradients of morphogen concentrations
        Pe : float
            Peclet number
        eta, alpha2, A_rd : float
            Reaction-diffusion parameters
        Mob_C1, Mob_C2 : float
            Mobilities for colloid type 1
        Mob2_C1, Mob2_C2 : float
            Mobilities for colloid type 2
        N, N1 : int
            Total particles and type 1 particles
        area_fraction : float
            Area fraction of particles
            
        Returns
        -------
        phi1, phi2 : np.ndarray
            Colloidal volume fractions
        """
        # Effective diffusion coefficient
        DNi = alpha2 * (Mob_C1 - Mob_C2 * eta * (1 + A_rd * eta) / A_rd) / Pe
        DN1 = DNi
        DN2 = DNi
        
        print(f"DNi (effective diffusion) = {DNi:.6e}")
        
        # Compute velocity fields
        Vx_dp1, Vy_dp1, Vx_dp2, Vy_dp2 = self._compute_velocity_fields(
            Dx_C1, Dx_C2, Dy_C1, Dy_C2,
            Mob_C1, Mob_C2, Mob2_C1, Mob2_C2
        )
        
        # Check continuity (divergence-free check)
        self._check_continuity(Vx_dp1, Vy_dp1, "Type 1")
        self._check_continuity(Vx_dp2, Vy_dp2, "Type 2")
        
        # Initialize volume fractions
        phi1_n = np.ones((self.nx + 2, self.ny + 2)) * (N1 / N) * area_fraction
        phi2_n = np.ones((self.nx + 2, self.ny + 2)) * ((N - N1) / N) * area_fraction
        
        phi1_n_1 = phi1_n.copy()
        phi2_n_1 = phi2_n.copy()
        
        # Time stepping
        for step in range(1, self.max_steps + 1):
            if step == 1:
                # Explicit Euler
                F1_n = self._compute_rhs(phi1_n, Vx_dp1, Vy_dp1, DN1)
                F2_n = self._compute_rhs(phi2_n, Vx_dp2, Vy_dp2, DN2)
                
                phi1_new = phi1_n + self.dt * F1_n
                phi2_new = phi2_n + self.dt * F2_n
            else:
                # Adams-Bashforth
                F1_n_1 = self._compute_rhs(phi1_n_1, Vx_dp1, Vy_dp1, DN1)
                F1_n = self._compute_rhs(phi1_n, Vx_dp1, Vy_dp1, DN1)
                F2_n_1 = self._compute_rhs(phi2_n_1, Vx_dp2, Vy_dp2, DN2)
                F2_n = self._compute_rhs(phi2_n, Vx_dp2, Vy_dp2, DN2)
                
                phi1_new = phi1_n + self.dt * (1.5 * F1_n - 0.5 * F1_n_1)
                phi2_new = phi2_n + self.dt * (1.5 * F2_n - 0.5 * F2_n_1)
                
                # Apply boundary conditions
                phi1_new = self._apply_periodic_bc(phi1_new)
                phi2_new = self._apply_periodic_bc(phi2_new)
                
                # Calculate error
                errorX = np.mean(np.abs(phi1_new - phi1_n))
                errorY = np.mean(np.abs(phi2_new - phi2_n))
                
                if step % 10000 == 0:
                    print(f"AD Step {step}: error1 = {errorX:.6e}, error2 = {errorY:.6e}")
                    
                    # Check conservation
                    sum1 = np.sum(phi1_new[1:-1, 1:-1])
                    sum2 = np.sum(phi2_new[1:-1, 1:-1])
                    IN1 = (N1 / N) * area_fraction * (self.nx * self.ny)
                    IN2 = ((N - N1) / N) * area_fraction * (self.nx * self.ny)
                    print(f"  Conservation: phi1 sum = {sum1:.6e} (target: {IN1:.6e})")
                    print(f"  Conservation: phi2 sum = {sum2:.6e} (target: {IN2:.6e})")
                
                # Check convergence
                if errorX < self.tol and errorY < self.tol:
                    print(f"Converged at step {step}")
                    break
            
            # Update
            phi1_n_1 = phi1_n.copy()
            phi2_n_1 = phi2_n.copy()
            phi1_n = phi1_new.copy()
            phi2_n = phi2_new.copy()
        
        # Extract interior points
        phi1 = phi1_n[1:self.nx+2, 1:self.ny+2]
        phi2 = phi2_n[1:self.nx+2, 1:self.ny+2]
        
        return phi1, phi2
    
    def _compute_velocity_fields(
        self,
        Dx_C1: np.ndarray,
        Dx_C2: np.ndarray,
        Dy_C1: np.ndarray,
        Dy_C2: np.ndarray,
        Mob_C1: float,
        Mob_C2: float,
        Mob2_C1: float,
        Mob2_C2: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute diffusiophoretic velocity fields."""
        Vx_dp1 = Mob_C1 * Dx_C1 + Mob_C2 * Dx_C2
        Vy_dp1 = Mob_C1 * Dy_C1 + Mob_C2 * Dy_C2
        Vx_dp2 = Mob2_C1 * Dx_C1 + Mob2_C2 * Dx_C2
        Vy_dp2 = Mob2_C1 * Dy_C1 + Mob2_C2 * Dy_C2
        
        return Vx_dp1, Vy_dp1, Vx_dp2, Vy_dp2
    
    def _check_continuity(self, Vx: np.ndarray, Vy: np.ndarray, label: str):
        """Check if velocity field is approximately divergence-free."""
        div_V = np.zeros_like(Vx)
        
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                div_V[i, j] = (
                    (Vx[i+1, j] - Vx[i-1, j]) / (2 * self.dx) +
                    (Vy[i, j+1] - Vy[i, j-1]) / (2 * self.dy)
                )
        
        max_div = np.max(np.abs(div_V[1:-1, 1:-1]))
        print(f"{label} velocity field - Max divergence: {max_div:.6e}")
    
    def _compute_rhs(
        self,
        phi: np.ndarray,
        Vx: np.ndarray,
        Vy: np.ndarray,
        D: float
    ) -> np.ndarray:
        """
        Compute RHS of advection-diffusion equation.
        
        ∂φ/∂t = -∇·(V*φ) + D*∇²φ
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        F = np.zeros_like(phi)
        
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                # Advection term: -∇·(V*φ) using upwind scheme
                if Vx[i, j] > 0:
                    advection_x = Vx[i, j] * (phi[i, j] - phi[i-1, j]) / dx
                else:
                    advection_x = Vx[i, j] * (phi[i+1, j] - phi[i, j]) / dx
                
                if Vy[i, j] > 0:
                    advection_y = Vy[i, j] * (phi[i, j] - phi[i, j-1]) / dy
                else:
                    advection_y = Vy[i, j] * (phi[i, j+1] - phi[i, j]) / dy
                
                advection = -(advection_x + advection_y)
                
                # Diffusion term: D*∇²φ using centered differences
                diffusion_x = D * (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) / dx**2
                diffusion_y = D * (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) / dy**2
                diffusion = diffusion_x + diffusion_y
                
                F[i, j] = advection + diffusion
        
        return F
    
    def _apply_periodic_bc(self, field: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions."""
        nx, ny = self.nx, self.ny
        
        # Corners
        field[0, 0] = field[nx, ny]
        field[0, ny+1] = field[nx, 1]
        field[nx+1, 0] = field[1, ny]
        field[nx+1, ny+1] = field[1, 1]
        
        # Edges
        field[0, 1:ny+1] = field[nx, 1:ny+1]
        field[nx+1, 1:ny+1] = field[1, 1:ny+1]
        field[1:nx+1, 0] = field[1:nx+1, ny]
        field[1:nx+1, ny+1] = field[1:nx+1, 1]
        
        return field
