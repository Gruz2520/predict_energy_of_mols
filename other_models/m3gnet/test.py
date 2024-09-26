import math
import sympy as sp

# Задаем символьные переменные
x = sp.symbols('x')

# Задаем многочлены f(x) и g(x)
fx = (sp.ln(7 + x**2, 1)) / 2
gx = (sp.sqrt(7)*sp.atan((sp.sqrt(7)*x)/7))/7

# Используем sp.Eq для сравнения двух выражений
eq = sp.Eq(fx, gx)

# Выводим результат
print(eq)

