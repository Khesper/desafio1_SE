import numpy as np

# Coeficientes de las ecuaciones
A = np.array([
    [52, 30, 18],
    [20, 50, 30],
    [25, 20, 55]
])

b = np.array([4800, 5810, 5690])

A_inv = np.linalg.inv(A)

x = np.dot(A_inv, b)

print("Matriz inversa de A:")
print(A_inv)
print()
# Resultados del sistema
b = np.array([4800, 5810, 5690])
x = np.linalg.solve(A, b)
print(x)
print()
print(f"Cantidad de material a transportar desde la cantera 1: {x[0]:.2f} metros cúbicos")
print(f"Cantidad de material a transportar desde la cantera 2: {x[1]:.2f} metros cúbicos")
print(f"Cantidad de material a transportar desde la cantera 3: {x[2]:.2f} metros cúbicos")
