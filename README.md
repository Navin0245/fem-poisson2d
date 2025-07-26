# Poisson Distribution using NumPy and Mesh Visualization

This project simulates the Poisson distribution using NumPy and visualizes the solution of the Poisson equation on a 2D mesh using finite element methods. Mesh files are handled using the `meshio` library.

## 📌 Objective

Solve the 2D Poisson equation:

\[
-\nabla^2 u = f \quad \text{in } \Omega
\]

with appropriate boundary conditions, by discretizing the domain into triangular elements, assembling the global stiffness matrix, and solving the resulting linear system.

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

Or manually install:

```bash
pip install numpy scipy matplotlib meshio
```

> Make sure to activate your virtual environment before running.

---

## 🚀 How to Run

From the root of the project directory, execute:

```bash
python src/poisson_distribution.py
```

## 📊 Output

* The program computes the finite element solution for the Poisson equation.
* It visualizes the result using `matplotlib` or exports the solution to a compatible VTK/mesh format for external visualization.

---

## 🧠 Concepts Used

* Finite Element Method (FEM)
* NumPy vectorized operations
* Sparse matrix assembly (SciPy)
* Boundary condition handling
* Mesh I/O via `meshio`

---

## 🛠️ Future Improvements

* Implement higher-order elements (e.g., quadratic)
* Support nonlinear source terms
* Add unit tests and benchmarking
* Create a GUI or interactive notebook

---

## 👨‍💻 Author

Navin C Chacko
Mechanical Engineer (FEA) | AI & ML Enthusiast
Email: [navincchacko@gmail.com](mailto:navincchacko@gmail.com)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

```