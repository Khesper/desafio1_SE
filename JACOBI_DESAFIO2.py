import numpy as np

def jacobi(A, b, x_init, tol=1e-5, max_iterations=100):
   
    n = len(b)
    x = x_init.copy()
    x_new = np.zeros_like(x)
    errores = []
    
    for k in range(max_iterations):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        error = np.max(np.abs(x_new - x))
        errores.append(error)
        
        print(f"Iteración {k + 1}: x = {x_new}, error = {error}")
        
        if error < tol:
            print("Convergencia alcanzada")
            break
        
        x = x_new.copy()
    
    return x_new, k + 1, errores


A = np.array([[52, 30, 18],
              [20, 50, 30],
              [25, 20, 55]])

b = np.array([4800, 5810, 5690])


x_init = np.array([0, 0, 0])

x_sol, num_iteraciones, errores = jacobi(A, b, x_init)

# Resultados finales
print("\nUltima Iteración:")
print(f"Número de iteraciones: {num_iteraciones}")
