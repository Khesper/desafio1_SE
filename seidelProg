import numpy as np

def gauss_seidel(A, b, x_init, tol=1e-4, max_iterations=100):
    
    n = len(b)
    x = x_init.copy()
    errores = []
    
    for k in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        
        # Calcular el error como la norma infinita (máximo error absoluto)
        error = np.max(np.abs(x - x_old))
        errores.append(error)
        
        # Formatear con 4 decimales
        x_formatted = [f"{val:.4f}" for val in x]
        error_formatted = f"{error:.4f}"
        
        print(f"Iteración {k + 1}: x = {x_formatted}, error = {error_formatted}")
        
        if error < tol:
            print("Convergencia alcanzada")
            break
    
    return x, k + 1, errores

A = np.array([[3, -0.1, -0.2],
              [0.1, 7, -0.3],
              [0.3, -0.2, 10]])

b = np.array([7.85, -19.3, 71.4])

# Valores iniciales de x
x_init = np.array([0, 0, 0])

# Ejecutar el método de Gauss-Seidel
x_sol, num_iteraciones, errores = gauss_seidel(A, b, x_init)

# Resultados finales
print("\nSolución final:")
x_sol_formatted = [f"{val:.4f}" for val in x_sol]
print(x_sol_formatted)
print(f"Número de iteraciones: {num_iteraciones}")
errores_formatted = [f"{err:.4f}" for err in errores]
print("Errores por iteración:", errores_formatted)
