import numpy as np
import matplotlib.pyplot as plt

def newton_divided_diff(x, y):
    """ Calcula la tabla de diferencias divididas de Newton """
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y  # Primera columna es y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])
    
    return coef[0, :]

def newton_interpolation(x_data, y_data, x):
    """ Evalúa el polinomio de Newton en los puntos x """
    coef = newton_divided_diff(x_data, y_data)
    n = len(x_data)
    
    y_interp = np.zeros_like(x)
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j-1])
            term += coef[j] * product
        y_interp[i] = term
    
    return y_interp

# Datos del ejercicio
F_data = np.array([50, 100, 150, 200])
epsilon_data = np.array([0.12, 0.35, 0.65, 1.05])

# Estimar la deformación para F = 125 N
F_target = 125
epsilon_target = newton_interpolation(F_data, epsilon_data, np.array([F_target]))
print(f"Deformación estimada para {F_target} N: {epsilon_target[0]:.4f} mm")

# Generar la gráfica
F_vals = np.linspace(min(F_data), max(F_data), 200)
epsilon_interp = newton_interpolation(F_data, epsilon_data, F_vals)

plt.figure(figsize=(8, 6))
plt.plot(F_data, epsilon_data, 'ro', label='Datos originales')
plt.plot(F_vals, epsilon_interp, 'b-', label='Interpolación de Newton')
plt.scatter(F_target, epsilon_target, color='green', label=f'Estimación ({F_target} N)')
plt.xlabel('Carga F (N)')
plt.ylabel('Deformación ε (mm)')
plt.title('Interpolación de Newton - Predicción de la Deformación')
plt.legend()
plt.grid(True)

# Guardar la gráfica
plt.savefig("newton_interpolacion_deformacion.png", dpi=300)
plt.show()
