from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'cambio-por-una-clave-segura'

DEFAULT_POINTS = [(1, 2), (2, 4), (3, 6.5), (4, 9), (5, 11.5)]

REGRESSION_TYPES = {
    'lineal': 'Regresión Lineal',
    'cuadratica': 'Regresión Cuadrática',
    'exponencial': 'Regresión Exponencial'
}


def parse_points(text):
    points = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        if ',' in line:
            parts = line.split(',')
        elif ' ' in line:
            parts = line.split()
        else:
            raise ValueError('Los puntos deben separarse por coma o espacio')
        if len(parts) < 2:
            raise ValueError('Cada línea debe contener dos valores: x e y')
        x = float(parts[0].strip())
        y = float(parts[1].strip())
        points.append((x, y))
    if len(points) < 2:
        raise ValueError('Se requieren al menos dos puntos para el ajuste')
    return points


def format_signed(value):
    sign = '+' if value >= 0 else '-'
    return f'{sign} {abs(value):.4f}'


def fmt(value):
    return f'{float(value):.4f}'


def fit_linear(points):
    if len(points) < 2:
        raise ValueError('Se requieren al menos dos puntos para el ajuste lineal')
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if len(np.unique(x)) < 2:
        raise ValueError('Se requieren al menos dos valores distintos de x')
    coef = np.polyfit(x, y, 1)
    a, b = coef
    def model(xv):
        return a * xv + b
    equation = f'y = {b:.4f} {format_signed(a)}x'
    n = len(points)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = (x**2).sum()
    sum_xy = (x * y).sum()
    denominator = n * sum_x2 - sum_x**2
    steps = [
        'Modelo: y = a + bx',
        f'n = {n}',
        f'Σx = {fmt(sum_x)}, Σy = {fmt(sum_y)}, Σx² = {fmt(sum_x2)}, Σxy = {fmt(sum_xy)}',
        'b = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²)',
        f'b = ({n}({fmt(sum_xy)}) - ({fmt(sum_x)})({fmt(sum_y)})) / ({n}({fmt(sum_x2)}) - ({fmt(sum_x)})²) = {fmt(a)}',
        'a = (Σy - bΣx) / n',
        f'a = ({fmt(sum_y)} - ({fmt(a)})({fmt(sum_x)})) / {n} = {fmt(b)}',
        f'Ecuación final: y = {fmt(b)} {format_signed(a)}x'
    ]
    return model, equation, steps


def fit_quadratic(points):
    if len(points) < 3:
        raise ValueError('Se requieren al menos tres puntos para el ajuste cuadrático')
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if len(np.unique(x)) < 3:
        raise ValueError('Se requieren al menos tres valores distintos de x')
    coef = np.polyfit(x, y, 2)
    a, b, c = coef
    def model(xv):
        return a * xv**2 + b * xv + c
    equation = f'y = {c:.4f} {format_signed(b)}x {format_signed(a)}x²'
    n = len(points)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = (x**2).sum()
    sum_x3 = (x**3).sum()
    sum_x4 = (x**4).sum()
    sum_xy = (x * y).sum()
    sum_x2y = ((x**2) * y).sum()
    steps = [
        'Modelo: y = a + bx + cx²',
        f'n = {n}',
        f'Σx = {fmt(sum_x)}, Σy = {fmt(sum_y)}, Σx² = {fmt(sum_x2)}',
        f'Σx³ = {fmt(sum_x3)}, Σx⁴ = {fmt(sum_x4)}, Σxy = {fmt(sum_xy)}, Σx²y = {fmt(sum_x2y)}',
        'Sistema de ecuaciones normales:',
        f'{n}a + {fmt(sum_x)}b + {fmt(sum_x2)}c = {fmt(sum_y)}',
        f'{fmt(sum_x)}a + {fmt(sum_x2)}b + {fmt(sum_x3)}c = {fmt(sum_xy)}',
        f'{fmt(sum_x2)}a + {fmt(sum_x3)}b + {fmt(sum_x4)}c = {fmt(sum_x2y)}',
        f'Al resolver el sistema: a = {fmt(c)}, b = {fmt(b)}, c = {fmt(a)}',
        f'Ecuación final: y = {fmt(c)} {format_signed(b)}x {format_signed(a)}x²'
    ]
    return model, equation, steps


def fit_exponential(points):
    if len(points) < 2:
        raise ValueError('Se requieren al menos dos puntos para el ajuste exponencial')
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if len(np.unique(x)) < 2:
        raise ValueError('Se requieren al menos dos valores distintos de x')
    if np.any(y <= 0):
        raise ValueError('Para regresión exponencial, todos los valores y deben ser positivos')
    logy = np.log(y)
    coef = np.polyfit(x, logy, 1)
    a, b = coef
    A = np.exp(b)
    def model(xv):
        return A * np.exp(a * xv)
    equation = f'y = {A:.4f} · e^({a:.4f}x)'
    n = len(points)
    sum_x = x.sum()
    sum_logy = logy.sum()
    sum_x2 = (x**2).sum()
    sum_x_logy = (x * logy).sum()
    intercept = b
    slope = a
    steps = [
        'Modelo: y = ae^(bx)',
        'Linealización: ln(y) = ln(a) + bx',
        f'n = {n}',
        f'Σx = {fmt(sum_x)}, Σln(y) = {fmt(sum_logy)}, Σx² = {fmt(sum_x2)}, Σxln(y) = {fmt(sum_x_logy)}',
        'b = (nΣxln(y) - ΣxΣln(y)) / (nΣx² - (Σx)²)',
        f'b = {fmt(slope)}',
        'ln(a) = (Σln(y) - bΣx) / n',
        f'ln(a) = {fmt(intercept)}, entonces a = e^{fmt(intercept)} = {fmt(A)}',
        f'Ecuación final: y = {fmt(A)} · e^({fmt(slope)}x)'
    ]
    return model, equation, steps


def create_plot(points, model, model_name, x_test=None, y_test=None):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    x_min, x_max = x.min(), x.max()
    xv = np.linspace(x_min - 1, x_max + 1, 300)
    yv = model(xv)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color='blue', label='Puntos originales')
    ax.plot(xv, yv, color='red', label=f'Ajuste {model_name}')
    if x_test is not None and y_test is not None:
        ax.scatter([x_test], [y_test], color='green', s=100, marker='X', label='Punto probado')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{model_name} y puntos originales')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


def get_regression(points, regression_type):
    if regression_type == 'lineal':
        return fit_linear(points)
    if regression_type == 'cuadratica':
        return fit_quadratic(points)
    if regression_type == 'exponencial':
        return fit_exponential(points)
    raise ValueError('Tipo de regresión no válido')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/regresion/<tipo>', methods=['GET', 'POST'])
def regresion(tipo):
    if tipo not in REGRESSION_TYPES:
        flash('Tipo de regresión no válido', 'danger')
        return redirect(url_for('home'))
    
    result_data = None
    sample_text = '\n'.join(f'{x}, {y}' for x, y in DEFAULT_POINTS)
    
    if request.method == 'POST':
        points_x = request.form.getlist('x[]')
        points_y = request.form.getlist('y[]')
        x_test_text = request.form.get('x_test', '')
        
        try:
            # Parse the points from individual x and y inputs
            points = []
            for x_str, y_str in zip(points_x, points_y):
                if x_str.strip() and y_str.strip():
                    x = float(x_str.strip())
                    y = float(y_str.strip())
                    points.append((x, y))
            
            if not points:
                points = DEFAULT_POINTS
            
            if len(points) < 2:
                raise ValueError('Se requieren al menos dos puntos para el ajuste')
            
            # Guardar los datos en sesión
            session['points_x'] = points_x
            session['points_y'] = points_y
            session['x_test'] = x_test_text
            session.modified = True
            
            model, equation, steps = get_regression(points, tipo)
            x_test = float(x_test_text) if x_test_text.strip() else None
            y_test = model(x_test) if x_test is not None else None
            plot_url = create_plot(points, model, REGRESSION_TYPES[tipo], x_test, y_test)
            result_data = {
                'regression_name': REGRESSION_TYPES[tipo],
                'equation': equation,
                'steps': steps,
                'points': points,
                'x_test': x_test,
                'y_test': y_test,
                'plot_url': plot_url,
                'regression_type': tipo
            }
        except Exception as e:
            flash(str(e), 'danger')

    # Cargar datos de sesión si existen
    points_x = session.get('points_x', [''])
    points_y = session.get('points_y', [''])
    x_test = session.get('x_test', '')

    return render_template('regresion.html',
                           tipo=tipo,
                           regression_name=REGRESSION_TYPES[tipo],
                           sample_text=sample_text,
                           result_data=result_data,
                           points_x=points_x,
                           points_y=points_y,
                           x_test=x_test,
                           regression_types=REGRESSION_TYPES)


if __name__ == '__main__':
    app.run(debug=True)
