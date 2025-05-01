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
    
    y_interp = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j-1])
            term += coef[j] * product
        y_interp[i] = term
    
    return y_interp

# Datos del ejercicio
V_data = np.array([10, 20, 30, 40, 50, 60])
Cd_data = np.array([0.32, 0.30, 0.28, 0.27, 0.26, 0.25])

# Estimar el coeficiente de arrastre para V = 35 m/s
V_target = 35
Cd_target = newton_interpolation(V_data, Cd_data, np.array([V_target]))
print(f"Coeficiente de arrastre estimado para {V_target} m/s: {Cd_target[0]:.4f}")

# Graficar la interpolación en el rango [10, 60]
V_vals = np.linspace(10, 60, 200)
Cd_interp = newton_interpolation(V_data, Cd_data, V_vals)

plt.figure(figsize=(8, 6))
plt.plot(V_data, Cd_data, 'ro', label='Datos experimentales')
plt.plot(V_vals, Cd_interp, 'b-', label='Interpolación de Newton')
plt.scatter(V_target, Cd_target, color='green', label=f'Estimación ({V_target} m/s)')
plt.xlabel('Velocidad del aire V (m/s)')
plt.ylabel('Coeficiente de arrastre $C_d$')
plt.title('Interpolación de Newton - Estimación del Coeficiente de Arrastre')
plt.legend()
plt.grid(True)

# Guardar la gráfica
plt.savefig("newton_interpolacion_coef_arrastre.png", dpi=300)
plt.show()
