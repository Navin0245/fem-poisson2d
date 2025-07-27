# Poisson 2D FEA Solver

A robust and efficient finite element analysis (FEA) solver for the 2D Poisson equation using linear triangular elements. This implementation provides a complete framework for solving elliptic partial differential equations with Dirichlet boundary conditions.

## 🔬 Mathematical Foundation

The solver addresses the 2D Poisson equation:

```
-∇ · (k∇u) = f(x,y)  in Ω
u = g(x,y)            on ∂Ω
```

Where:
- `u(x,y)` is the unknown solution field
- `k` is the thermal conductivity coefficient
- `f(x,y)` is the source term
- `g(x,y)` represents Dirichlet boundary conditions
- `Ω` is the computational domain

## ✨ Key Features

- **Linear Triangular Elements**: High-quality FEA implementation with proper shape functions
- **Robust Mesh Generation**: Automatic Delaunay triangulation for arbitrary domains
- **Sparse Matrix Operations**: Efficient assembly and solving using SciPy sparse matrices
- **Boundary Condition Handling**: Robust Dirichlet boundary condition enforcement
- **Visualization Suite**: Comprehensive plotting and VTK export capabilities
- **Convergence Analysis**: Built-in convergence study tools

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/POISSON_DISTRIBUTION_USING_NUMPY.git
cd POISSON_DISTRIBUTION_USING_NUMPY

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.poisson_distribution import Poisson2D

# Create solver for unit square domain
solver = Poisson2D(a=1.0, n=30, conductivity=1.0)

# Solve the complete problem
solver.solve_complete()

# Access solution
temperature_field = solver.U
mesh_points = solver.points
```

### Custom Source Function

```python
from src.poisson_distribution import Poisson2D
import numpy as np

class CustomPoisson(Poisson2D):
    def source_function(self, x, y):
        """Custom source term: sinusoidal heating"""
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def boundary_condition(self, x, y):
        """Custom boundary conditions"""
        return 0.0  # Homogeneous Dirichlet

# Use custom solver
solver = CustomPoisson(a=1.0, n=40)
solver.solve_complete()
```

## 📊 Output Examples

The solver generates comprehensive visualizations:

### Mesh Visualization
- Node distribution and element connectivity
- Boundary node identification
- Mesh quality assessment

### Solution Fields
- Filled contour plots with customizable colormaps
- Contour line plots with labels
- 3D surface representations

### Analysis Tools
- Source function visualization
- Convergence study plots
- Residual analysis

### Convergence Study

```python
from src.poisson_distribution import Poisson2D

# Perform mesh convergence analysis
solver = Poisson2D(a=1.0, n=20)
solver.solve_complete()

# Study convergence with different mesh densities
n_values = [10, 15, 20, 25, 30, 40]
solver.convergence_study(n_values)
```

### VTK Export for ParaView

```python
# Export results for advanced visualization in paraview
solver.export_to_vtk("outputs/temperature_field.vtk")
```

### Custom Material Properties

```python
# Variable conductivity
solver = Poisson2D(a=2.0, n=50, conductivity=2.5)
```

## 📁 Project Structure

```
POISSON_DISTRIBUTION_USING_NUMPY/
├── .venv/                     # Virtual environment
├── .vs/                       # Visual Studio settings
├── .vscode/                   # VSCode configuration
├── LICENSE                    # Project license
├── outputs/                   # Generated results and visualizations
│   ├── convergence_study.png  # Mesh convergence analysis
│   ├── mesh_overview.png      # Mesh visualization
│   ├── poisson_solution.vtk   # VTK export for ParaView
│   ├── solution.png           # FEM solution plots
│   └── source_function.png    # Source function visualization
├── src/
│   └── poisson_distribution.py # Main FEA solver implementation
├── tests/
│   └── test.py                # Unit tests and validation
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## 🔧 Dependencies

```txt
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
meshio>=5.0.0
```

### Development Setup

```bash
# Create development environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test.py

# Run the main solver
python src/poisson_distribution.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.