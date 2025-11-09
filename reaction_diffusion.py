"""
Reaction-Diffusion Solver for Brusselator Model
================================================

Solves the reaction-diffusion equations using finite difference method (FDM)
to generate Turing patterns that serve as a blueprint for particle motion.
"""

import numpy as np
from typing import Tuple


class ReactionDiffusionSolver:
    """
    Finite Difference Method solver for reaction-diffusion equations
    using the Brusselator model.
    
    Brusselator reactions:
        dX/dt = Da_c * (A - (B+1)*X + X^2*Y) + D_X * ∇^2X
        dY/dt = Da_c * (B*X - X^2*Y) + D_Y * ∇^2Y
    
    where D_Y = D_rd * D_X (D_rd is relative diffusion coefficient)
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        size_x: float,
        size_y: float,
        dt: float,
        max_steps: int,
        tol: float,
        A: float,
        D_rd: float,
        mu: float,
        Da_c: float,
        am_noise: float
    ):
        """
        Initialize the reaction-diffusion solver.
        
        Parameters
        ----------
        nx, ny : int
            Grid points in x and y directions
        size_x, size_y : float
            Domain size
        dt : float
            Time step
        max_steps : int
            Maximum number of time steps
        tol : float
            Convergence tolerance
        A : float
            Brusselator parameter A
        D_rd : float
            Relative diffusion coefficient (D_Y / D_X)
        mu : float
            Control parameter
        Da_c : float
            Damköhler number
        am_noise : float
            Amplitude of initial noise
        """
        self.nx = nx
        self.ny = ny
        self.size_x = size_x
        self.size_y = size_y
        self.dt = dt
        self.max_steps = max_steps
        self.tol = tol
        self.A = A
        self.D_rd = D_rd
        self.mu = mu
        self.Da_c = Da_c
        self.am_noise = am_noise
        
        # Grid spacing
        self.dx = size_x / nx
        self.dy = size_y / ny
        
        # Calculate derived parameters
        self.eta = 1.0 / np.sqrt(D_rd)
        Bc = (1 + A * self.eta) ** 2
        self.B = Bc * (1 + mu)
        
        # Create mesh
        self.mesh_x = np.linspace(0, size_x, nx + 1)
        self.mesh_y = np.linspace(0, size_y, ny + 1)
        
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                            np.ndarray, np.ndarray, float]:
        """
        Solve reaction-diffusion equations to steady state.
        
        Returns
        -------
        C1, C2 : np.ndarray
            Concentration fields at steady state (nx+1 x ny+1)
        Dx_C1, Dx_C2, Dy_C1, Dy_C2 : np.ndarray
            Gradients of concentrations (nx+2 x ny+2 with BC)
        alpha2 : float
            Perturbation amplitude
        """
        # Initialize concentration fields
        X_n = np.zeros((self.nx + 1, self.ny + 1))
        Y_n = np.zeros((self.nx + 1, self.ny + 1))
        X_n_1 = np.zeros((self.nx + 1, self.ny + 1))
        Y_n_1 = np.zeros((self.nx + 1, self.ny + 1))
        
        # Initial conditions with noise
        noise = np.random.uniform(0, 1, (self.nx + 1, self.ny + 1))
        X_n[:, :] = self.A
        Y_n[:, :] = self.B / self.A + self.am_noise * noise
        
        # Time stepping
        for step in range(1, self.max_steps + 1):
            if step == 1:
                # Explicit Euler for first step
                Fx_n, Fy_n = self._compute_rhs(X_n, Y_n)
                X_new = X_n + self.dt * Fx_n
                Y_new = Y_n + self.dt * Fy_n
            else:
                # Adams-Bashforth 2nd order
                Fx_n_1, Fy_n_1 = self._compute_rhs(X_n_1, Y_n_1)
                Fx_n, Fy_n = self._compute_rhs(X_n, Y_n)
                
                X_new = X_n + self.dt * (1.5 * Fx_n - 0.5 * Fx_n_1)
                Y_new = Y_n + self.dt * (1.5 * Fy_n - 0.5 * Fy_n_1)
                
                # Apply periodic boundary conditions
                X_new = self._apply_periodic_bc(X_new)
                Y_new = self._apply_periodic_bc(Y_new)
                
                # Calculate error
                errorX = np.mean(np.abs(X_new - X_n))
                errorY = np.mean(np.abs(Y_new - Y_n))
                
                if step % 1000 == 0:
                    print(f"Step {step}: errorX = {errorX:.6e}, errorY = {errorY:.6e}")
            
            # Update for next iteration
            X_n_1 = X_n.copy()
            Y_n_1 = Y_n.copy()
            X_n = X_new.copy()
            Y_n = Y_new.copy()
        
        # Store steady-state concentrations
        C1 = X_n.copy()
        C2 = Y_n.copy()
        
        # Calculate perturbation amplitude (alpha)
        mean_C1 = np.mean(C1)
        alpha2 = np.sqrt(np.mean((C1 - mean_C1) ** 2))
        print(f"Alpha (perturbation amplitude) = {alpha2:.6f}")
        
        # Calculate concentration gradients
        Dx_C1, Dx_C2, Dy_C1, Dy_C2 = self._compute_gradients(C1, C2)
        
        return C1, C2, Dx_C1, Dx_C2, Dy_C1, Dy_C2, alpha2
    
    def _compute_rhs(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute right-hand side of reaction-diffusion equations.
        
        Parameters
        ----------
        X, Y : np.ndarray
            Current concentration fields
            
        Returns
        -------
        Fx, Fy : np.ndarray
            Time derivatives
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        # Initialize derivatives
        DX_x = np.zeros((nx + 1, ny + 1))
        DX_y = np.zeros((nx + 1, ny + 1))
        DY_x = np.zeros((nx + 1, ny + 1))
        DY_y = np.zeros((nx + 1, ny + 1))
        
        # Second derivatives using finite differences with periodic BC
        for i in range(nx + 1):
            for j in range(ny + 1):
                # x-direction second derivative
                if 0 < i < nx:
                    DX_x[i, j] = (X[i+1, j] - 2*X[i, j] + X[i-1, j]) / dx**2
                    DY_x[i, j] = (Y[i+1, j] - 2*Y[i, j] + Y[i-1, j]) / dx**2
                elif i == 0:  # Periodic BC
                    DX_x[i, j] = (X[i+1, j] - 2*X[i, j] + X[nx-1, j]) / dx**2
                    DY_x[i, j] = (Y[i+1, j] - 2*Y[i, j] + Y[nx-1, j]) / dx**2
                else:  # i == nx, periodic BC
                    DX_x[i, j] = (X[1, j] - 2*X[i, j] + X[i-1, j]) / dx**2
                    DY_x[i, j] = (Y[1, j] - 2*Y[i, j] + Y[i-1, j]) / dx**2
                
                # y-direction second derivative
                if 0 < j < ny:
                    DX_y[i, j] = (X[i, j+1] - 2*X[i, j] + X[i, j-1]) / dy**2
                    DY_y[i, j] = (Y[i, j+1] - 2*Y[i, j] + Y[i, j-1]) / dy**2
                elif j == 0:  # Periodic BC
                    DX_y[i, j] = (X[i, j+1] - 2*X[i, j] + X[i, ny-1]) / dy**2
                    DY_y[i, j] = (Y[i, j+1] - 2*Y[i, j] + Y[i, ny-1]) / dy**2
                else:  # j == ny, periodic BC
                    DX_y[i, j] = (X[i, 1] - 2*X[i, j] + X[i, j-1]) / dy**2
                    DY_y[i, j] = (Y[i, 1] - 2*Y[i, j] + Y[i, j-1]) / dy**2
        
        # Reaction terms
        Rc1 = self.Da_c * (self.A - (self.B + 1) * X + X**2 * Y)
        Rc2 = self.Da_c * (self.B * X - X**2 * Y)
        
        # Combined RHS
        Fx = DX_x + DX_y + Rc1
        Fy = self.D_rd * (DY_x + DY_y) + Rc2
        
        return Fx, Fy
    
    def _apply_periodic_bc(self, field: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions to a field."""
        # Corner
        field[-1, -1] = field[0, 0]
        
        # Edges
        field[-1, :-1] = field[0, :-1]  # Right edge
        field[:-1, -1] = field[:-1, 0]  # Top edge
        
        return field
    
    def _compute_gradients(
        self, 
        C1: np.ndarray, 
        C2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute concentration gradients using centered differences.
        
        Parameters
        ----------
        C1, C2 : np.ndarray
            Concentration fields (nx+1 x ny+1)
            
        Returns
        -------
        Dx_C1, Dx_C2, Dy_C1, Dy_C2 : np.ndarray
            Gradients with boundary conditions (nx+2 x ny+2)
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        # Initialize gradient arrays with padding for BC
        Dx_C1 = np.zeros((nx + 2, ny + 2))
        Dx_C2 = np.zeros((nx + 2, ny + 2))
        Dy_C1 = np.zeros((nx + 2, ny + 2))
        Dy_C2 = np.zeros((nx + 2, ny + 2))
        
        # Compute gradients in interior
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                # Centered differences
                Dx_C1[i, j] = (
                    C1[i, j] + C1[i, j-1] - C1[i-1, j] - C1[i-1, j-1]
                ) / (2.0 * dx)
                Dx_C2[i, j] = (
                    C2[i, j] + C2[i, j-1] - C2[i-1, j] - C2[i-1, j-1]
                ) / (2.0 * dx)
                Dy_C1[i, j] = (
                    C1[i, j] + C1[i-1, j] - C1[i, j-1] - C1[i-1, j-1]
                ) / (2.0 * dy)
                Dy_C2[i, j] = (
                    C2[i, j] + C2[i-1, j] - C2[i, j-1] - C2[i-1, j-1]
                ) / (2.0 * dy)
        
        # Apply periodic boundary conditions
        self._apply_gradient_bc(Dx_C1)
        self._apply_gradient_bc(Dx_C2)
        self._apply_gradient_bc(Dy_C1)
        self._apply_gradient_bc(Dy_C2)
        
        return Dx_C1, Dx_C2, Dy_C1, Dy_C2
    
    def _apply_gradient_bc(self, grad: np.ndarray):
        """Apply periodic boundary conditions to gradient field."""
        nx, ny = self.nx, self.ny
        
        # Corners
        grad[0, 0] = grad[nx, ny]
        grad[0, ny+1] = grad[nx, 1]
        grad[nx+1, 0] = grad[1, ny]
        grad[nx+1, ny+1] = grad[1, 1]
        
        # Edges (i direction)
        for i in range(nx + 2):
            grad[i, 0] = grad[i, ny]
            grad[i, ny+1] = grad[i, 1]
        
        # Edges (j direction)
        for j in range(ny + 2):
            grad[0, j] = grad[nx, j]
            grad[nx+1, j] = grad[1, j]


def visualize_turing_pattern(C1: np.ndarray, C2: np.ndarray, 
                             output_file: str = "turing_pattern.png"):
    """
    Visualize the Turing pattern.
    
    Parameters
    ----------
    C1, C2 : np.ndarray
        Concentration fields
    output_file : str
        Output filename for the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(C1.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title('Species X (C1)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(C2.T, origin='lower', cmap='plasma', aspect='auto')
    axes[1].set_title('Species Y (C2)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"Saved Turing pattern visualization to {output_file}")
