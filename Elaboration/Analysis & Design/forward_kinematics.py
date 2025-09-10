import numpy as np
from scipy.optimize import least_squares
import itertools
import math

# --- 1) Define tus rangos y pasos para las 3 variables libres ---
# Ejemplo (ajusta min,max,num según tu caso)
f1_vals = np.linspace(f1_min, f1_max, n1)
f2_vals = np.linspace(f2_min, f2_max, n2)
f3_vals = np.linspace(f3_min, f3_max, n3)

# --- 2) Preallocate arrays para resultados ---
# soluciones: shape (n1,n2,n3,4) para las 4 variables dependientes
solutions = np.full((n1, n2, n3, 4), np.nan, dtype=float)
res_norm = np.full((n1, n2, n3), np.nan, dtype=float)
success = np.zeros((n1, n2, n3), dtype=bool)
message = np.empty((n1, n2, n3), dtype=object)

# --- 3) Define la función de residuo ---
# x: vector de 4 variables dependientes
# free_vars: (f1,f2,f3) — valores fijos para este punto de la malla
def residuals(x, free_vars):
    f1, f2, f3 = free_vars
    # --- AQUI coloca tus 4 ecuaciones; ejemplo genérico:
    r0 = eq0(x, f1, f2, f3)
    r1 = eq1(x, f1, f2, f3)
    r2 = eq2(x, f1, f2, f3)
    r3 = eq3(x, f1, f2, f3)
    return np.array([r0, r1, r2, r3])
    raise NotImplementedError("Sustituye por tus ecuaciones")

# --- 4) Sweep (ordenado para permitir warm starts) ---
# inicial_guess global (si no hay vecino)
global_guess = np.array([g0, g1, g2, g3], dtype=float)

for i, f1 in enumerate(f1_vals):
    for j, f2 in enumerate(f2_vals):
        for k, f3 in enumerate(f3_vals):
            # intenta warm-start con vecino anterior en k, luego j, luego i
            x0 = None
            if k > 0 and not np.isnan(solutions[i,j,k-1,0]):
                x0 = solutions[i,j,k-1]
            elif j > 0 and not np.isnan(solutions[i,j-1,k,0]):
                x0 = solutions[i,j-1,k]
            elif i > 0 and not np.isnan(solutions[i-1,j,k,0]):
                x0 = solutions[i-1,j,k]
            else:
                x0 = global_guess

            fun = lambda x: residuals(x, (f1,f2,f3))

            try:
                res = least_squares(fun, x0, method='lm', xtol=1e-10, ftol=1e-10, gtol=1e-10)
            except Exception as e:
                success[i,j,k] = False
                message[i,j,k] = f"error: {e}"
                continue

            solutions[i,j,k,:] = res.x
            res_norm[i,j,k] = np.linalg.norm(res.fun)
            success[i,j,k] = res.success
            message[i,j,k] = res.message

# --- 5) Guarda resultados a disco (opcional) ---
np.savez_compressed("grid_solutions.npz",
                    solutions=solutions,
                    res_norm=res_norm,
                    success=success,
                    f1_vals=f1_vals, f2_vals=f2_vals, f3_vals=f3_vals)
