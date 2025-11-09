"""
GPU-Accelerated Reaction-Diffusion Solver
==========================================

CuPy-based implementation for CUDA acceleration on GPU.
Drop-in replacement for reaction_diffusion.py
"""

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("✗ CuPy not found - falling back to CPU")

import numpy as np
from typing import Tuple


class ReactionDiffusionSolverGPU:
    """
    GPU-accelerated Brusselator solver using CuPy.
    
    For A100 GPU, this can be 50-100x faster than CPU for large grids.
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
        
        # Derived parameters
        self.eta = 1.0 / cp.sqrt(D_rd)
        Bc = (1 + A * self.eta) ** 2
        self.B = Bc * (1 + mu)
        
        # Pre-compute stencil coefficients for Laplacian
        self.dx2_inv = 1.0 / (self.dx ** 2)
        self.dy2_inv = 1.0 / (self.dy ** 2)
        
        print(f"GPU RD Solver initialized: {nx}x{ny} grid")
        if GPU_AVAILABLE:
            print(f"  Device: {cp.cuda.Device().compute_capability}")
            mem_info = cp.cuda.Device().mem_info
            print(f"  GPU Memory: {mem_info[1]/1e9:.1f} GB total, "
                  f"{mem_info[0]/1e9:.1f} GB free")
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                            np.ndarray, np.ndarray, float]:
        """
        Solve reaction-diffusion equations on GPU.
        
        Returns arrays on CPU (as NumPy arrays).
        """
        # Initialize on GPU
        X_n = cp.zeros((self.nx + 1, self.ny + 1), dtype=cp.float64)
        Y_n = cp.zeros((self.nx + 1, self.ny + 1), dtype=cp.float64)
        X_n_1 = cp.zeros((self.nx + 1, self.ny + 1), dtype=cp.float64)
        Y_n_1 = cp.zeros((self.nx + 1, self.ny + 1), dtype=cp.float64)
        
        # Initial conditions with noise (on GPU)
        noise = cp.random.uniform(0, 1, (self.nx + 1, self.ny + 1))
        X_n[:, :] = self.A
        Y_n[:, :] = self.B / self.A + self.am_noise * noise
        
        # Time stepping loop
        for step in range(1, self.max_steps + 1):
            if step == 1:
                # Explicit Euler
                Fx_n, Fy_n = self._compute_rhs_gpu(X_n, Y_n)
                X_new = X_n + self.dt * Fx_n
                Y_new = Y_n + self.dt * Fy_n
            else:
                # Adams-Bashforth 2nd order
                Fx_n_1, Fy_n_1 = self._compute_rhs_gpu(X_n_1, Y_n_1)
                Fx_n, Fy_n = self._compute_rhs_gpu(X_n, Y_n)
                
                X_new = X_n + self.dt * (1.5 * Fx_n - 0.5 * Fx_n_1)
                Y_new = Y_n + self.dt * (1.5 * Fy_n - 0.5 * Fy_n_1)
                
                # Apply periodic BC
                X_new = self._apply_periodic_bc_gpu(X_new)
                Y_new = self._apply_periodic_bc_gpu(Y_new)
                
                # Calculate error (on GPU)
                errorX = float(cp.mean(cp.abs(X_new - X_n)))
                errorY = float(cp.mean(cp.abs(Y_new - Y_n)))
                
                if step % 1000 == 0:
                    print(f"Step {step}: errorX = {errorX:.6e}, errorY = {errorY:.6e}")
            
            # Update
            X_n_1 = X_n.copy()
            Y_n_1 = Y_n.copy()
            X_n = X_new
            Y_n = Y_new
        
        # Transfer final results to CPU
        C1 = cp.asnumpy(X_n)
        C2 = cp.asnumpy(Y_n)
        
        # Calculate alpha on GPU then transfer
        mean_C1 = float(cp.mean(X_n))
        alpha2 = float(cp.sqrt(cp.mean((X_n - mean_C1) ** 2)))
        print(f"Alpha (perturbation amplitude) = {alpha2:.6f}")
        
        # Compute gradients on GPU
        Dx_C1, Dx_C2, Dy_C1, Dy_C2 = self._compute_gradients_gpu(X_n, Y_n)
        
        # Transfer to CPU
        Dx_C1 = cp.asnumpy(Dx_C1)
        Dx_C2 = cp.asnumpy(Dx_C2)
        Dy_C1 = cp.asnumpy(Dy_C1)
        Dy_C2 = cp.asnumpy(Dy_C2)
        
        return C1, C2, Dx_C1, Dx_C2, Dy_C1, Dy_C2, alpha2
    
    def _compute_rhs_gpu(self, X: cp.ndarray, Y: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Compute RHS using GPU-accelerated operations.
        
        This is the performance-critical kernel.
        """
        nx, ny = self.nx, self.ny
        
        # Allocate output arrays on GPU
        Fx = cp.zeros_like(X)
        Fy = cp.zeros_like(Y)
        
        # Compute Laplacians using vectorized operations
        # This is much faster on GPU than explicit loops
        
        # Central differences for interior points
        # d²X/dx²
        DX_x = cp.zeros_like(X)
        DX_x[1:-1, :] = (X[2:, :] - 2*X[1:-1, :] + X[:-2, :]) * self.dx2_inv
        # Periodic BC for boundaries
        DX_x[0, :] = (X[1, :] - 2*X[0, :] + X[-2, :]) * self.dx2_inv
        DX_x[-1, :] = (X[1, :] - 2*X[-1, :] + X[-2, :]) * self.dx2_inv
        
        # d²X/dy²
        DX_y = cp.zeros_like(X)
        DX_y[:, 1:-1] = (X[:, 2:] - 2*X[:, 1:-1] + X[:, :-2]) * self.dy2_inv
        # Periodic BC
        DX_y[:, 0] = (X[:, 1] - 2*X[:, 0] + X[:, -2]) * self.dy2_inv
        DX_y[:, -1] = (X[:, 1] - 2*X[:, -1] + X[:, -2]) * self.dy2_inv
        
        # Same for Y with different diffusion coefficient
        DY_x = cp.zeros_like(Y)
        DY_x[1:-1, :] = (Y[2:, :] - 2*Y[1:-1, :] + Y[:-2, :]) * self.dx2_inv
        DY_x[0, :] = (Y[1, :] - 2*Y[0, :] + Y[-2, :]) * self.dx2_inv
        DY_x[-1, :] = (Y[1, :] - 2*Y[-1, :] + Y[-2, :]) * self.dx2_inv
        
        DY_y = cp.zeros_like(Y)
        DY_y[:, 1:-1] = (Y[:, 2:] - 2*Y[:, 1:-1] + Y[:, :-2]) * self.dy2_inv
        DY_y[:, 0] = (Y[:, 1] - 2*Y[:, 0] + Y[:, -2]) * self.dy2_inv
        DY_y[:, -1] = (Y[:, 1] - 2*Y[:, -1] + Y[:, -2]) * self.dy2_inv
        
        # Reaction terms (all vectorized on GPU)
        X2Y = X * X * Y  # Only compute once
        Rc1 = self.Da_c * (self.A - (self.B + 1) * X + X2Y)
        Rc2 = self.Da_c * (self.B * X - X2Y)
        
        # Combined RHS
        Fx = DX_x + DX_y + Rc1
        Fy = self.D_rd * (DY_x + DY_y) + Rc2
        
        return Fx, Fy
    
    def _apply_periodic_bc_gpu(self, field: cp.ndarray) -> cp.ndarray:
        """Apply periodic boundary conditions on GPU."""
        # Corner
        field[-1, -1] = field[0, 0]
        # Edges (vectorized)
        field[-1, :-1] = field[0, :-1]
        field[:-1, -1] = field[:-1, 0]
        return field
    
    def _compute_gradients_gpu(
        self, 
        C1: cp.ndarray, 
        C2: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Compute gradients on GPU using centered differences.
        """
        nx, ny = self.nx, self.ny
        
        # Allocate with padding for BC
        Dx_C1 = cp.zeros((nx + 2, ny + 2), dtype=cp.float64)
        Dx_C2 = cp.zeros((nx + 2, ny + 2), dtype=cp.float64)
        Dy_C1 = cp.zeros((nx + 2, ny + 2), dtype=cp.float64)
        Dy_C2 = cp.zeros((nx + 2, ny + 2), dtype=cp.float64)
        
        # Vectorized gradient computation on interior
        # Using centered differences
        dx_inv = 1.0 / (2.0 * self.dx)
        dy_inv = 1.0 / (2.0 * self.dy)
        
        # Compute on interior points (vectorized)
        Dx_C1[1:-1, 1:-1] = (
            C1[1:, 1:] + C1[1:, :-1] - C1[:-1, 1:] - C1[:-1, :-1]
        ) * dx_inv
        Dx_C2[1:-1, 1:-1] = (
            C2[1:, 1:] + C2[1:, :-1] - C2[:-1, 1:] - C2[:-1, :-1]
        ) * dx_inv
        Dy_C1[1:-1, 1:-1] = (
            C1[1:, 1:] + C1[:-1, 1:] - C1[1:, :-1] - C1[:-1, :-1]
        ) * dy_inv
        Dy_C2[1:-1, 1:-1] = (
            C2[1:, 1:] + C2[:-1, 1:] - C2[1:, :-1] - C2[:-1, :-1]
        ) * dy_inv
        
        # Apply periodic BC
        self._apply_gradient_bc_gpu(Dx_C1, nx, ny)
        self._apply_gradient_bc_gpu(Dx_C2, nx, ny)
        self._apply_gradient_bc_gpu(Dy_C1, nx, ny)
        self._apply_gradient_bc_gpu(Dy_C2, nx, ny)
        
        return Dx_C1, Dx_C2, Dy_C1, Dy_C2
    
    def _apply_gradient_bc_gpu(self, grad: cp.ndarray, nx: int, ny: int):
        """Apply periodic BC to gradient field on GPU."""
        # Corners
        grad[0, 0] = grad[nx, ny]
        grad[0, ny+1] = grad[nx, 1]
        grad[nx+1, 0] = grad[1, ny]
        grad[nx+1, ny+1] = grad[1, 1]
        
        # Edges (vectorized)
        grad[:, 0] = grad[:, ny]
        grad[:, ny+1] = grad[:, 1]
        grad[0, :] = grad[nx, :]
        grad[nx+1, :] = grad[1, :]


# Backward compatibility: if CuPy not available, use CPU version
if not GPU_AVAILABLE:
    print("Warning: CuPy not available. Import from reaction_diffusion.py instead.")
    from reaction_diffusion import ReactionDiffusionSolver as ReactionDiffusionSolverGPU


def benchmark_gpu_vs_cpu(nx=256, ny=256, steps=1000):
    """
    Benchmark GPU vs CPU performance.
    """
    import time
    
    params = {
        'nx': nx, 'ny': ny,
        'size_x': 32.0, 'size_y': 32.0,
        'dt': 0.00003, 'max_steps': steps,
        'tol': 1e-12,
        'A': 4.5, 'D_rd': 8.0, 'mu': 0.04,
        'Da_c': 1.0, 'am_noise': 0.02
    }
    
    print("="*60)
    print(f"Benchmarking: {nx}x{ny} grid, {steps} steps")
    print("="*60)
    
    if GPU_AVAILABLE:
        print("\n[GPU] Testing...")
        solver_gpu = ReactionDiffusionSolverGPU(**params)
        start = time.time()
        C1_gpu, C2_gpu, *_ = solver_gpu.solve()
        gpu_time = time.time() - start
        print(f"GPU Time: {gpu_time:.2f} seconds")
    
    print("\n[CPU] Testing...")
    from reaction_diffusion import ReactionDiffusionSolver
    solver_cpu = ReactionDiffusionSolver(**params)
    start = time.time()
    C1_cpu, C2_cpu, *_ = solver_cpu.solve()
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.2f} seconds")
    
    if GPU_AVAILABLE:
        speedup = cpu_time / gpu_time
        print(f"\n{'='*60}")
        print(f"Speedup: {speedup:.1f}x faster on GPU")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Quick test
    benchmark_gpu_vs_cpu(nx=256, ny=256, steps=1000)
