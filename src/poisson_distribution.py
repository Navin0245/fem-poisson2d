# Corrected Poisson 2D FEA Solver using Linear Triangular Elements
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import meshio

class Poisson2D:
    def __init__(self, a, n, conductivity=1.0):
        """
        Initialize Poisson 2D solver
        
        Parameters:
        a: domain size (square domain [0,a] x [0,a])
        n: number of points per side
        conductivity: thermal conductivity coefficient (k in -∇·(k∇u) = f)
        """
        self.a = a
        self.n = n
        self.k = conductivity  # thermal conductivity
        self.output_folder = "outputs"
        self.X, self.Y = self.create_grid()
        self.points, self.tri = self.generate_mesh()
        self.boundary_nodes = self.identify_boundary_nodes()
        
    def create_grid(self):
        """Create structured grid points"""
        x = np.linspace(0, self.a, self.n)
        y = np.linspace(0, self.a, self.n)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def generate_mesh(self):
        """Generate triangular mesh using Delaunay triangulation"""
        # Flatten the 2D meshgrid into 1D arrays
        x_flat = self.X.flatten()
        y_flat = self.Y.flatten()
        points = np.column_stack((x_flat, y_flat))
        
        # Create triangulation using Delaunay
        tri = Delaunay(points)
        
        return points, tri

    def identify_boundary_nodes(self):
        """Identify boundary nodes more robustly"""
        eps = 1e-12
        boundary_mask = (
            (np.abs(self.points[:, 0] - 0) < eps) |          # left boundary
            (np.abs(self.points[:, 0] - self.a) < eps) |     # right boundary
            (np.abs(self.points[:, 1] - 0) < eps) |          # bottom boundary
            (np.abs(self.points[:, 1] - self.a) < eps)       # top boundary
        )
        return np.where(boundary_mask)[0]

    def plot_mesh(self):
        """Plot the mesh with nodes and elements"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot nodes
        ax1.plot(self.X, self.Y, 'o', markersize=2, color='blue')
        ax1.set_title('Mesh Nodes')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot mesh with elements
        triangulation = mtri.Triangulation(self.points[:, 0], self.points[:, 1], 
                                         self.tri.simplices)
        ax2.triplot(triangulation, color='gray', linewidth=0.5)
        ax2.plot(self.points[:, 0], self.points[:, 1], 'o', color='red', markersize=1)
        
        # Highlight boundary nodes
        boundary_points = self.points[self.boundary_nodes]
        ax2.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', 
                color='green', markersize=2, label='Boundary nodes')
        
        ax2.set_title('Triangular Mesh with Elements')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        self._save_figure('mesh_overview.png')

    def source_function(self, x, y):
        """
        Define source function f(x,y) for the Poisson equation -∇²u = f
        Default: Gaussian source centered at domain center
        """
        xc, yc = self.a/2, self.a/2  # center coordinates
        sigma = 0.1 * self.a  # width parameter
        return np.exp(-((x - xc)**2 + (y - yc)**2) / sigma**2)

    def boundary_condition(self, x, y):
        """
        Define Dirichlet boundary conditions u = g on boundary
        Default: homogeneous boundary conditions (u = 0)
        """
        return 0.0

    def element_stiffness_matrix(self, coords):
        """
        Compute element stiffness matrix for linear triangular element
        
        For Poisson equation: Ke_ij = ∫∫ k * (∇Ni · ∇Nj) dA
        
        Parameters:
        coords: array of shape (3,2) containing coordinates of triangle vertices
        
        Returns:
        Ke: 3x3 element stiffness matrix
        """
        # Extract coordinates
        x1, y1 = coords[0]
        x2, y2 = coords[1] 
        x3, y3 = coords[2]
        
        # Compute area using cross product
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        if area <= 1e-12:
            print(f"Warning: Degenerate triangle with area {area}")
            return np.zeros((3, 3))
        
        # Shape function derivatives (constant for linear triangular elements)
        # ∇N1 = (1/2A) * [y2-y3, x3-x2]
        # ∇N2 = (1/2A) * [y3-y1, x1-x3]  
        # ∇N3 = (1/2A) * [y1-y2, x2-x1]
        
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        
        # B matrix contains shape function derivatives
        # B = [b1 b2 b3; c1 c2 c3] / (2*Area)
        B = np.array([b, c]) / (2 * area)
        
        # Element stiffness matrix: Ke = k * Area * B^T * B
        # This integrates ∇Ni · ∇Nj over the element
        Ke = self.k * area * (B.T @ B)
        
        return Ke

    def element_load_vector(self, coords):
        """
        Compute element load vector for linear triangular element
        
        fe_i = ∫∫ f(x,y) * Ni(x,y) dA
        
        Using centroid rule for integration (exact for linear f over triangle)
        """
        # Get triangle centroid
        xc = np.mean(coords[:, 0])
        yc = np.mean(coords[:, 1])
        
        # Compute area
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        # Evaluate source function at centroid
        f_val = self.source_function(xc, yc)
        
        # For linear elements with constant source over element,
        # load is distributed equally among nodes
        fe = (f_val * area / 3) * np.ones(3)
        
        return fe

    def assemble_system(self):
        """
        Assemble global stiffness matrix and load vector
        """
        num_nodes = len(self.points)
        
        # Initialize global matrices (use LIL format for efficient assembly)
        K_global = lil_matrix((num_nodes, num_nodes))
        F_global = np.zeros(num_nodes)
        
        print(f"Assembling system for {len(self.tri.simplices)} elements...")
        
        # Loop over all elements
        for elem_idx, element in enumerate(self.tri.simplices):
            # Get element coordinates
            coords = self.points[element]
            
            # Compute element matrices
            Ke = self.element_stiffness_matrix(coords)
            fe = self.element_load_vector(coords)
            
            # Assemble into global system
            for i in range(3):
                global_i = element[i]
                F_global[global_i] += fe[i]
                
                for j in range(3):
                    global_j = element[j]
                    K_global[global_i, global_j] += Ke[i, j]
        
        # Convert to CSR format for efficient solving
        self.K = K_global.tocsr()
        self.F = F_global
        
        print(f"System assembled: {num_nodes} DOFs")

    def apply_boundary_conditions(self):
        """
        Apply Dirichlet boundary conditions using elimination method
        """
        print(f"Applying boundary conditions to {len(self.boundary_nodes)} nodes...")
        
        # Convert to LIL format for efficient modification
        K_bc = self.K.tolil()
        F_bc = self.F.copy()
        
        for node in self.boundary_nodes:
            x, y = self.points[node]
            bc_value = self.boundary_condition(x, y)
            
            # Elimination method: set row to identity, RHS to BC value
            K_bc[node, :] = 0
            K_bc[node, node] = 1
            F_bc[node] = bc_value
        
        # Convert back to CSR
        self.K = K_bc.tocsr()
        self.F = F_bc

    def solve_system(self):
        """
        Solve the linear system Ku = F
        """
        print("Solving linear system...")
        
        # Solve using sparse direct solver
        self.U = spsolve(self.K, self.F)
        
        # Check solution quality
        residual = self.K @ self.U - self.F
        residual_norm = np.linalg.norm(residual)
        
        print(f"Solution completed")
        print(f"Residual norm: {residual_norm:.3e}")
        print(f"Max solution value: {np.max(self.U):.6f}")
        print(f"Min solution value: {np.min(self.U):.6f}")

    def plot_solution(self):
        """
        Plot the numerical solution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Contour plot
        triangulation = mtri.Triangulation(self.points[:, 0], self.points[:, 1], 
                                         self.tri.simplices)
        
        # Filled contour plot
        contour = ax1.tricontourf(triangulation, self.U, levels=20, cmap='viridis')
        cbar1 = fig.colorbar(contour, ax=ax1)
        cbar1.set_label('Solution u(x, y)')
        ax1.set_title('FEM Solution - Filled Contours')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.axis('equal')
        
        # Line contour plot
        contour_lines = ax2.tricontour(triangulation, self.U, levels=15, colors='black', linewidths=0.5)
        ax2.clabel(contour_lines, inline=True, fontsize=8)
        surface = ax2.tricontourf(triangulation, self.U, levels=20, cmap='plasma', alpha=0.7)
        cbar2 = fig.colorbar(surface, ax=ax2)
        cbar2.set_label('Solution u(x, y)')
        ax2.set_title('FEM Solution - Contour Lines')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.axis('equal')
        
        plt.tight_layout()
        self._save_figure('solution.png')

    def plot_source_function(self):
        """
        Plot the source function for reference
        """
        plt.figure(figsize=(10, 8))
        
        # Create fine grid for smooth plotting
        x_fine = np.linspace(0, self.a, 100)
        y_fine = np.linspace(0, self.a, 100)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        Z_fine = self.source_function(X_fine, Y_fine)
        
        contour = plt.contourf(X_fine, Y_fine, Z_fine, levels=20, cmap='coolwarm')
        cbar = plt.colorbar(contour)
        cbar.set_label('Source function f(x, y)')
        plt.title('Source Function Distribution')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.axis('equal')
        
        self._save_figure('source_function.png')

    def export_to_vtk(self, filename=None):
        """
        Export solution to VTK format for visualization in ParaView
        """
        if filename is None:
            filename = os.path.join(self.output_folder, "poisson_solution.vtk")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create 3D points (z=0 for 2D problem)
        points_3d = np.column_stack([self.points, np.zeros(len(self.points))])
        
        # Create cells
        cells = [("triangle", self.tri.simplices)]
        
        # Create mesh with solution data
        mesh = meshio.Mesh(
            points=points_3d,
            cells=cells,
            point_data={
                "temperature": self.U,
                "source": [self.source_function(p[0], p[1]) for p in self.points]
            }
        )
        
        # Write mesh
        mesh.write(filename)
        print(f"VTK file exported to: {filename}")

    def convergence_study(self, n_values):
        """
        Perform convergence study by solving on different mesh sizes
        """
        errors = []
        h_values = []
        
        print("Performing convergence study...")
        
        for n in n_values:
            print(f"Solving for n = {n}...")
            
            # Create temporary solver
            temp_solver = Poisson2D(self.a, n, self.k)
            temp_solver.assemble_system()
            temp_solver.apply_boundary_conditions()
            temp_solver.solve_system()
            
            # Compute characteristic mesh size
            h = self.a / (n - 1)
            h_values.append(h)
            
            # For this example, we'll use the L2 norm of the solution as error metric
            # In practice, you'd compare against analytical solution if available
            error = np.linalg.norm(temp_solver.U) * h  # Scale by mesh size
            errors.append(error)
            
            print(f"  h = {h:.4f}, error metric = {error:.6f}")
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.loglog(h_values, errors, 'o-', label='FEM Solution')
        plt.loglog(h_values, np.array(h_values)**2 * errors[0]/h_values[0]**2, 
                  '--', label='h² (theoretical)')
        plt.xlabel('Mesh size h')
        plt.ylabel('Error metric')
        plt.title('Convergence Study')
        plt.legend()
        plt.grid(True)
        self._save_figure('convergence_study.png')

    def _save_figure(self, filename):
        """Save figure to output folder"""
        os.makedirs(self.output_folder, exist_ok=True)
        filepath = os.path.join(self.output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
        plt.close()

    def solve_complete(self):
        """
        Complete solution workflow
        """
        print("="*60)
        print("POISSON 2D FEA SOLVER")
        print("="*60)
        print(f"Domain: [0, {self.a}] × [0, {self.a}]")
        print(f"Grid points: {self.n} × {self.n}")
        print(f"Total nodes: {len(self.points)}")
        print(f"Total elements: {len(self.tri.simplices)}")
        print(f"Boundary nodes: {len(self.boundary_nodes)}")
        print("-"*60)
        
        # Solve the problem
        self.assemble_system()
        self.apply_boundary_conditions()
        self.solve_system()
        
        # Generate plots
        self.plot_mesh()
        self.plot_source_function()
        self.plot_solution()
        self.export_to_vtk()
        
        print("-"*60)
        print("Solution completed successfully!")
        print("="*60)

# Example usage and demonstration
if __name__ == "__main__":
    # Create and solve problem
    solver = Poisson2D(a=1.0, n=30, conductivity=1.0)
    solver.solve_complete()
    
    # Optional: Run convergence study
    print("\nRunning convergence study...")
    solver.convergence_study([10, 15, 20, 25, 30])
    
    print("\nAll outputs saved to 'outputs/' folder")