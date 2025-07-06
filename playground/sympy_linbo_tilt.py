# %%
import sympy

t, n, theta = sympy.symbols("t n theta")

# %%

r = sympy.asin(sympy.sin(theta) / n)

a = t / sympy.cos(r)
dely = a*sympy.cos()
