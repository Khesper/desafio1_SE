import numpy as np

def gauss_seidel(A, b, x_init, tol=1e-5, max_iterations=100):
    
    n = len(b)
    x = x_init.copy().astype(float)  
    errores = []
    
    for k in range(max_iterations):
        x_old = x.copy()  
        
        for i in range(n):
            
            s1 = sum(A[i][j] * x[j] for j in range(i))  
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))  
            x[i] = (b[i] - s1 - s2) / A[i][i]
        
        error = np.max(np.abs(x - x_old))
        errores.append(error)
        
        x_formatted = [f"{xi:.6f}" for xi in x]
        error_formatted = f"{error:.6f}"
        
        print(f"Iteración {k + 1}: x = {x_formatted}, error = {error_formatted}")
        
        if error < tol:
            print("Convergencia alcanzada")
            break
    
    return x, k + 1, errores

A = np.array([[52.0, 30.0, 18.0],
              [20.0, 50.0, 30.0],
              [25.0, 20.0, 55.0]])

b = np.array([4800.0, 5810.0, 5690.0])

x_init = np.array([0.0, 0.0, 0.0])

x_sol, num_iteraciones, errores = gauss_seidel(A, b, x_init)

print("\nSolución final:")
x_sol_formatted = [f"{xi:.6f}" for xi in x_sol]
print(x_sol_formatted)
print(f"Número de iteraciones: {num_iteraciones}")
errores_formatted = [f"{err:.6f}" for err in errores]
print("Errores por iteración:", errores_formatted)
