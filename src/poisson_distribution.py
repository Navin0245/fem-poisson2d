# importing essential libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import meshio

class Poisson2D:
    def __init__(self, a, n):
        self.a = a
        self.n = n
        self.output_folder = "outputs"
        self.X, self.Y = self.quad()
        self.points, self.tri = self.generate_mesh()
        self.K, self.F = self.assemble_stiffness_matrix()

    def quad(self):
        x = np.linspace(0, self.a, self.n)
        y = np.linspace(0, self.a, self.n)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def plot_node(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.X, self.Y, 'o', markersize=0.5)
        plt.title('Mesh Nodes')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        self._save_figure('mesh_nodes.png')

    def generate_mesh(self):
        # Flatten the 2D meshgrid into 1D arrays
        X = self.X
        Y = self.Y
        x_flat = X.flatten()
        y_flat = Y.flatten()
        points = np.vstack((x_flat, y_flat)).T
        # Create triangulation using Delaunay
        tri = Delaunay(points)
        triangles = tri.simplices  # element connectivity
        plt.figure(figsize=(8, 8))
        triangulation = mtri.Triangulation(x_flat, y_flat, triangles)
        plt.triplot(triangulation, color='gray', linewidth=0.5)
        plt.plot(self.X, self.Y, 'o', color='red', markersize=0.5)  # Optional: overlay points
        plt.title('Mesh Grid with Elements')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.axis('equal')
        self._save_figure('mesh_elements.png')
        return points, tri

    def boundary_region(self):
        eps = 1e-10
        condition = (
            np.isclose(self.points[:, 0], 0, atol=eps) |
            np.isclose(self.points[:, 0], self.a, atol=eps) |
            np.isclose(self.points[:, 1], 0, atol=eps) |
            np.isclose(self.points[:, 1], self.a, atol=eps)
        )
        bnd = self.points[condition]
        load_region = self.points[~condition]
        return bnd, load_region
    
    def dirichilet_bc(self,bnd, value = 0):
        bc = {tuple(node): value for node in bnd}
        return bc

    def plot_boundary(self, bnd, load_region):
        plt.figure(figsize=(8, 8))
        plt.scatter(bnd[:, 0], bnd[:, 1], color='red', label='Boundary Points', s=1.5)
        plt.scatter(load_region[:, 0], load_region[:, 1], color='blue', label='Load Points', s=1.5)
        plt.title('Boundary and Load Regions')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc = 'best')
        self._save_figure('boundary_load_regions.png')    
            
    def _save_figure(self, filename):
        # Create folder if not exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        save_path = os.path.join(self.output_folder, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
        plt.close()

    def triangle_stiffness(self, coords, A):
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        if A <= 0:
            raise ValueError("Degenerate triangle with non-positive area")
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.vstack((b, c)) / (2 * A)
        return A * (B.T @ B)

    def assemble_stiffness_matrix(self):
        N = self.points.shape[0]
        
        # Step 1: Assemble using LIL (fast for setting values)
        K = lil_matrix((N, N))
        F = np.zeros(N)

        for idx, triangle in enumerate(self.tri.simplices):
            i, j, k = triangle
            p1, p2, p3 = self.points[i], self.points[j], self.points[k]

            # Precompute area once here
            matrix = np.array([
                [p2[0] - p1[0], p3[0] - p1[0]],
                [p2[1] - p1[1], p3[1] - p1[1]]
            ])
            area = 0.5 * np.abs(np.linalg.det(matrix))

            # Use area in both stiffness and load computation
            coords = [p1, p2, p3]
            
            Ke = self.triangle_stiffness(coords, area)
            fe = self.load_vector(coords, area)

            # Assemble
            for m, a in enumerate(triangle):
                F[a] += fe[m]
                for n, b in enumerate(triangle):
                    K[a, b] += Ke[m, n]

        # Step 2: Convert to CSR after all modifications
        return K.tocsr(), F
    
    def load_vector(self, coords, area):
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        xc = (x1 + x2 + x3) / 3
        yc = (y1 + y2 + y3) / 3
        f_val = np.exp(-((xc - self.a/2)**2 + (yc - self.a/2)**2) / (0.1 * self.a)**2)
        return f_val * area / 3 * np.ones(3)  # equal share to all 3 nodes

    def apply_dirichlet_bc(self):
        bnd, _ = self.boundary_region()
        tree = cKDTree(self.points)
        _, bnd_indices = tree.query(bnd, k=1)

        self.K = self.K.tolil()
        for node in bnd_indices:
            self.K[node, :] = 0
            self.K[node, node] = 1
            self.F[node] = 0
        self.K = self.K.tocsr()

    def solve(self):
        self.U = spsolve(self.K, self.F)
        residual = self.K @ self.U - self.F
        print(f"Residual norm: {np.linalg.norm(residual):.3e}")

    def plot_solution(self):
        plt.figure(figsize=(8, 8))
        contour = plt.tricontourf(self.points[:, 0], self.points[:, 1],
                                self.tri.simplices, self.U,
                                cmap='rainbow', levels=20)
        cbar = plt.colorbar(contour)
        cbar.set_label('Solution u(x, y)')
        plt.title("FEM Solution to Poisson Equation")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.axis('equal')
        self._save_figure('solution.png')
    
    def export_to_vtk(self, filename="outputs/solution.vtk"):
        # Convert triangles to VTK format (cells)
        cells = [("triangle", self.tri.simplices)]
        
        # Create mesh with meshio
        mesh = meshio.Mesh(
            points=self.points,
            cells=cells,
            point_data={"u": self.U}  # Solution at nodes
        )
        
        # Save the mesh
        mesh.write(filename)
        print(f"VTK file saved to: {filename}")







mesh = Poisson2D(a=1.0, n=100)
mesh.plot_node()
bnd, load = mesh.boundary_region()
mesh.plot_boundary(bnd, load)
mesh.apply_dirichlet_bc()
mesh.solve()
mesh.plot_solution()
mesh.export_to_vtk()