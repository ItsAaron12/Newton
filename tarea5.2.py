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
T_data = np.array([200, 250, 300, 350, 400])
efficiency_data = np.array([30, 35, 40, 46, 53])

# Estimar la eficiencia para T = 275 °C
T_target = 275
efficiency_target = newton_interpolation(T_data, efficiency_data, np.array([T_target]))
print(f"Eficiencia estimada para {T_target} °C: {efficiency_target[0]:.4f} %")

# Graficar la interpolación en el rango [200, 400]
T_vals = np.linspace(200, 400, 200)
efficiency_interp = newton_interpolation(T_data, efficiency_data, T_vals)

plt.figure(figsize=(8, 6))
plt.plot(T_data, efficiency_data, 'ro', label='Datos experimentales')
plt.plot(T_vals, efficiency_interp, 'b-', label='Interpolación de Newton')
plt.scatter(T_target, efficiency_target, color='green', label=f'Estimación ({T_target} °C)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Eficiencia (%)')
plt.title('Interpolación de Newton - Estimación de la Eficiencia de un Motor Térmico')
plt.legend()
plt.grid(True)

# Guardar la gráfica
plt.savefig("newton_interpolacion_eficiencia_motor.png", dpi=300)
plt.show()
