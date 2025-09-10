# Código para resolver el sistema no lineal y encontrar relaciones funcionales
# gamma1(theta1, theta2, gamma4), gamma2(theta1, theta2, gamma4), 
# gamma3(theta1, theta2, gamma4), l1(theta1, theta2, gamma4)
#
# Ejecuta 180 iteraciones para cada parámetro libre en el rango [0, π]

import numpy as np
try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy no está disponible. Usando implementación alternativa.")
    SCIPY_AVAILABLE = False

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# --- Implementación alternativa simple de Levenberg-Marquardt ---
def simple_lm_solver(fun, x0, jac=None, max_iter=100, tol=1e-10):
    """
    Implementación simple de Levenberg-Marquardt como fallback
    """
    x = np.array(x0, dtype=float)
    lambda_lm = 1e-3
    
    for i in range(max_iter):
        f = fun(x)
        residual_norm = np.linalg.norm(f)
        
        if residual_norm < tol:
            return {'x': x, 'success': True, 'nfev': i+1, 'fun': f}
        
        if jac is not None:
            J = jac(x)
        else:
            # Diferencias finitas numéricas
            h = 1e-8
            J = np.zeros((len(f), len(x)))
            for j in range(len(x)):
                x_plus = x.copy()
                x_plus[j] += h
                f_plus = fun(x_plus)
                J[:, j] = (f_plus - f) / h
        
        # Levenberg-Marquardt step
        JTJ = J.T @ J
        JTf = J.T @ f
        
        try:
            # Intentar paso Gauss-Newton
            delta = np.linalg.solve(JTJ, -JTf)
            x_new = x + delta
            f_new = fun(x_new)
            
            if np.linalg.norm(f_new) < residual_norm:
                x = x_new
                lambda_lm *= 0.1
            else:
                # Usar regularización LM
                delta = np.linalg.solve(JTJ + lambda_lm * np.eye(len(x)), -JTf)
                x = x + delta
                lambda_lm *= 10
                
        except np.linalg.LinAlgError:
            # Si la matriz es singular, usar regularización
            delta = np.linalg.solve(JTJ + lambda_lm * np.eye(len(x)), -JTf)
            x = x + delta
            lambda_lm *= 10
    
    return {'x': x, 'success': False, 'nfev': max_iter, 'fun': fun(x)}

# --- Definición de residuales y jacobiano ---
def residuals_free(free_vars, mask_fixed, fixed_vals, theta1, theta2, gamma4):
    """
    free_vars: vector con solo las variables libres en el orden [g1,g2,g3,l1] filtrado por mask_fixed
    mask_fixed: boolean array length 4; True = variable está fija, False = variable libre
                orden: [gamma1, gamma2, gamma3, l1]
    fixed_vals: array length 4 con valores (válidos donde mask_fixed True, ignóralos donde False)
    theta1, theta2, gamma4: parámetros (constantes)
    """
    # Reconstruir vector completo x = [g1,g2,g3,l1]
    x = np.array(fixed_vals, dtype=float)
    x[~mask_fixed] = free_vars  # fill free positions
    g1, g2, g3, l1 = x.tolist()
    
    # Precomputados
    ctheta = -55.6 * (np.cos(theta1) + np.cos(theta2))
    stheta = -55.6 * (np.sin(theta1) + np.sin(theta2))
    cg4 = np.cos(gamma4)
    sg4 = np.sin(gamma4)
    
    # Ecuaciones f1..f4
    f1 = ctheta + 190.9*np.cos(g1) + 304.7*np.cos(g2) - 214.5*np.cos(g3) - 47.4*cg4 - 72.0
    f2 = stheta - 190.9*np.sin(g1) + 304.7*np.sin(g2) - 214.5*np.sin(g3) + 47.4*sg4
    f3 = 47.4*np.cos(g1) + l1*np.cos(g2) - 214.5*np.cos(g3) - 47.4*cg4
    f4 = -47.4*np.sin(g1) + l1*np.sin(g2) - 214.5*np.sin(g3) + 47.4*sg4
    
    return np.array([f1, f2, f3, f4])

def jacobian_free(free_vars, mask_fixed, fixed_vals, theta1, theta2, gamma4):
    """
    Jacobiano analítico de las 4 ecuaciones con respecto a las 4 variables (g1,g2,g3,l1),
    pero devuelve solo las columnas correspondientes a las variables libres.
    """
    x = np.array(fixed_vals, dtype=float)
    x[~mask_fixed] = free_vars
    g1, g2, g3, l1 = x.tolist()
    
    # Derivadas parciales de f1..f4 respecto a [g1,g2,g3,l1]
    df1_dg1 = -190.9 * np.sin(g1)
    df1_dg2 = -304.7 * np.sin(g2)
    df1_dg3 =  214.5 * np.sin(g3)
    df1_dl1 = 0.0
    
    df2_dg1 = -190.9 * np.cos(g1)
    df2_dg2 =  304.7 * np.cos(g2)
    df2_dg3 = -214.5 * np.cos(g3)  # Corregido: era sin, debe ser -cos
    df2_dl1 = 0.0
    
    df3_dg1 = -47.4 * np.sin(g1)
    df3_dg2 = -l1 * np.sin(g2)
    df3_dg3 =  214.5 * np.sin(g3)
    df3_dl1 =  np.cos(g2)
    
    df4_dg1 = -47.4 * np.cos(g1)
    df4_dg2 =  l1 * np.cos(g2)
    df4_dg3 = -214.5 * np.cos(g3)
    df4_dl1 =  np.sin(g2)
    
    J_full = np.array([
        [df1_dg1, df1_dg2, df1_dg3, df1_dl1],
        [df2_dg1, df2_dg2, df2_dg3, df2_dl1],
        [df3_dg1, df3_dg2, df3_dg3, df3_dl1],
        [df4_dg1, df4_dg2, df4_dg3, df4_dl1],
    ])  # shape (4,4)
    
    # Retornar solo columnas correspondientes a variables libres, en orden
    return J_full[:, ~mask_fixed]

# --- Función para resolver un caso individual ---
def solve_single_case(theta1, theta2, gamma4, x0_guess=None, method='lm', max_attempts=5):
    """
    Resuelve un caso individual del sistema no lineal
    """
    fixed_mask = [False, False, False, False]  # Todas las variables son libres
    fixed_vals = [0.0, 0.0, 0.0, 100.0]  # Valores por defecto
    
    if x0_guess is None:
        x0_guess = np.array([0.5, 0.5, 0.5, 150.0])  # Guess inicial por defecto
    
    mask = np.array(fixed_mask, dtype=bool)
    
    # Diferentes valores iniciales para probar
    initial_guesses = [
        np.array([0.5, 0.5, 0.5, 150.0]),
        np.array([0.1, 0.1, 0.1, 100.0]),
        np.array([1.0, 1.0, 1.0, 200.0]),
        np.array([np.pi/4, np.pi/4, np.pi/4, 180.0]),
        np.array([theta1/2, theta2/2, gamma4/2, 120.0])
    ]
    
    for attempt in range(max_attempts):
        try:
            if attempt < len(initial_guesses):
                x0_current = initial_guesses[attempt]
            else:
                # Añadir pequeña perturbación aleatoria para intentos adicionales
                x0_current = x0_guess + np.random.normal(0, 0.1, size=len(x0_guess))
            
            fun = lambda fv: residuals_free(fv, mask, fixed_vals, theta1, theta2, gamma4)
            jac_fun = lambda fv: jacobian_free(fv, mask, fixed_vals, theta1, theta2, gamma4)
            
            if SCIPY_AVAILABLE:
                res = least_squares(fun, x0_current, jac=jac_fun, method=method,
                                  ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
                if res.success and np.linalg.norm(res.fun) < 1e-6:
                    return res.x, True
            else:
                res = simple_lm_solver(fun, x0_current, jac=jac_fun, max_iter=500, tol=1e-8)
                if res['success'] and np.linalg.norm(res['fun']) < 1e-6:
                    return res['x'], True
                    
        except Exception as e:
            continue
    
    return None, False

# --- Función principal para generar datos paramétricos ---
def generate_parametric_data(n_iterations=180):
    """
    Genera datos para n_iterations de cada parámetro en [0, π]
    """
    # Rangos para cada parámetro
    theta_range = np.linspace(0, np.pi, n_iterations)
    gamma_range = np.linspace(0, np.pi, n_iterations)
    
    data = []
    total_cases = len(theta_range) * len(theta_range) * len(gamma_range)
    successful_cases = 0
    
    print(f"Iniciando análisis paramétrico con {total_cases} casos...")
    print("Progreso: ", end="", flush=True)
    
    case_count = 0
    progress_interval = max(1, total_cases // 100)  # Mostrar progreso cada 1%
    
    for i, theta1 in enumerate(theta_range):
        for j, theta2 in enumerate(theta_range):
            for k, gamma4 in enumerate(gamma_range):
                case_count += 1
                
                # Mostrar progreso
                if case_count % progress_interval == 0:
                    progress_pct = (case_count / total_cases) * 100
                    print(f"{progress_pct:.1f}% ", end="", flush=True)
                
                # Resolver el sistema para estos parámetros
                solution, success = solve_single_case(theta1, theta2, gamma4)
                
                if success:
                    gamma1, gamma2, gamma3, l1 = solution
                    
                    # Validar que la solución es física (ej. l1 > 0)
                    if l1 > 0 and l1 < 1000:  # Rango razonable para l1
                        data.append({
                            'theta1': theta1,
                            'theta2': theta2, 
                            'gamma4': gamma4,
                            'gamma1': gamma1,
                            'gamma2': gamma2,
                            'gamma3': gamma3,
                            'l1': l1
                        })
                        successful_cases += 1
    
    print(f"\nAnálisis completado: {successful_cases}/{total_cases} casos exitosos ({100*successful_cases/total_cases:.1f}%)")
    
    if successful_cases == 0:
        raise ValueError("No se encontraron soluciones válidas en ningún caso")
    
    return pd.DataFrame(data)

# --- Función para ajustar modelos polinomiales ---
def fit_polynomial_models(df, degree=3):
    """
    Ajusta modelos polinomiales para cada variable de salida
    """
    # Variables de entrada
    X = df[['theta1', 'theta2', 'gamma4']].values
    
    # Variables de salida
    variables = ['gamma1', 'gamma2', 'gamma3', 'l1']
    models = {}
    
    print("\nAjustando modelos polinomiales...")
    for var in variables:
        y = df[var].values
        
        # Crear pipeline con características polinomiales
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
            ('linear', LinearRegression())
        ])
        
        # Ajustar modelo
        model.fit(X, y)
        
        # Calcular R²
        r2_score = model.score(X, y)
        
        models[var] = {
            'model': model,
            'r2_score': r2_score
        }
        
        print(f"Modelo para {var}: R² = {r2_score:.6f}")
    
    return models

# --- Función para generar expresiones simbólicas ---
def generate_symbolic_expressions(models, degree=3):
    """
    Genera las expresiones simbólicas de los polinomios ajustados
    """
    def get_feature_names(degree):
        """Genera los nombres de las características polinomiales"""
        names = ['1']  # Término independiente
        
        # Términos lineales
        vars = ['θ₁', 'θ₂', 'γ₄']
        for var in vars:
            names.append(var)
        
        # Términos cuadráticos
        if degree >= 2:
            # Términos cuadráticos puros
            for var in vars:
                names.append(f'{var}²')
            # Términos cruzados
            for i in range(len(vars)):
                for j in range(i+1, len(vars)):
                    names.append(f'{vars[i]}·{vars[j]}')
        
        # Términos cúbicos
        if degree >= 3:
            # Términos cúbicos puros
            for var in vars:
                names.append(f'{var}³')
            # Términos cruzados de grado 3
            for i in range(len(vars)):
                for j in range(len(vars)):
                    if i != j:
                        names.append(f'{vars[i]}²·{vars[j]}')
            # Término triple
            names.append('θ₁·θ₂·γ₄')
        
        return names
    
    feature_names = get_feature_names(degree)
    
    print("\n" + "="*80)
    print("EXPRESIONES FUNCIONALES AJUSTADAS")
    print("="*80)
    
    for var_name, model_info in models.items():
        model = model_info['model']
        r2 = model_info['r2_score']
        coefficients = model.named_steps['linear'].coef_
        
        # Construir la expresión
        terms = []
        for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
            if abs(coef) > 1e-10:  # Solo incluir coeficientes significativos
                if feature == '1':
                    terms.append(f"{coef:.6f}")
                else:
                    if coef >= 0 and len(terms) > 0:
                        terms.append(f" + {coef:.6f}·{feature}")
                    else:
                        terms.append(f"{coef:.6f}·{feature}")
        
        expression = ''.join(terms)
        if not expression:
            expression = "0"
        
        print(f"\n{var_name}(θ₁, θ₂, γ₄) = {expression}")
        print(f"R² = {r2:.6f}")
        print("-" * 60)
    
    return models

# --- Función para validar soluciones ---
def validate_solutions(df, sample_size=10):
    """
    Valida algunas soluciones verificando que satisfacen el sistema original
    """
    print(f"\nValidando {sample_size} soluciones aleatorias...")
    
    # Seleccionar muestra aleatoria
    sample_df = df.sample(n=min(sample_size, len(df)))
    
    max_error = 0
    for idx, row in sample_df.iterrows():
        theta1, theta2, gamma4 = row['theta1'], row['theta2'], row['gamma4']
        gamma1, gamma2, gamma3, l1 = row['gamma1'], row['gamma2'], row['gamma3'], row['l1']
        
        # Ahora todas las variables están fijadas; no hay libres
        mask = np.array([True, True, True, True], dtype=bool)
        fixed_vals = [gamma1, gamma2, gamma3, l1]
        residuals = residuals_free(np.array([]), mask, fixed_vals, theta1, theta2, gamma4)
        error = np.linalg.norm(residuals)
        max_error = max(max_error, error)
        
        if error > 1e-5:
            print(f"  Advertencia: Error elevado = {error:.2e}")
    
    print(f"Validación completada. Error máximo: {max_error:.2e}")
    return max_error < 1e-4


# --- Función principal ---
def main():
    print("Iniciando análisis paramétrico del sistema de cinemática...")
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy available: {SCIPY_AVAILABLE}")
    
    # Generar datos paramétricos
    try:
        print(f"\nGenerando datos con resolución reducida para prueba inicial...")
        df = generate_parametric_data(n_iterations=30)  # Empezar con menos puntos
        
        print(f"\nDatos generados: {len(df)} puntos válidos")
        print("\nEstadísticas de los datos:")
        print(df.describe())
        
        # Validar soluciones
        if not validate_solutions(df):
            print("Advertencia: Algunas soluciones pueden no ser precisas")
        
        # Ajustar modelos polinomiales
        models = fit_polynomial_models(df, degree=2)  # Empezar con grado 2
        
        # Generar expresiones simbólicas
        generate_symbolic_expressions(models, degree=2)
        
        # Guardar datos si es posible
        try:
            df.to_csv('parametric_data.csv', index=False)
            print("\nDatos guardados en 'parametric_data.csv'")
        except Exception as e:
            print(f"\nNo se pudieron guardar los datos: {e}")
        
        return df, models
        
    except Exception as e:
        print(f"Error en el análisis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# --- Ejecución principal ---
if __name__ == "__main__":
    df, models = main()