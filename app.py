from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


def parse_number(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"El valor de {field_name} debe ser un numero valido.")


def calculate_linear_regression(points, x_test=None):
    n = len(points)
    if n < 2:
        raise ValueError("Debes ingresar al menos 2 puntos para calcular la regresion lineal.")

    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_x2 = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denominator = (n * sum_x2) - (sum_x * sum_x)
    if denominator == 0:
        raise ValueError("Los valores de X no pueden ser todos iguales.")

    b = ((n * sum_xy) - (sum_x * sum_y)) / denominator
    a = (sum_y - (b * sum_x)) / n

    y_mean = sum_y / n
    predictions = [a + b * x for x in xs]
    ss_total = sum((y - y_mean) ** 2 for y in ys)
    ss_residual = sum((y - y_pred) ** 2 for y, y_pred in zip(ys, predictions))
    r2 = 1 if ss_total == 0 else 1 - (ss_residual / ss_total)

    result = {
        "a": a,
        "b": b,
        "r2": r2,
        "equation": f"y = {a:.6f} + {b:.6f}x",
        "points": points,
    }

    if x_test is not None:
        result["x_test"] = x_test
        result["y_pred"] = a + b * x_test

    return result


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/regresion/lineal")
@app.route("/regresion/<tipo>")
def regresion(tipo="lineal"):
    return render_template("regresion.html")


@app.route("/api/regresion-lineal", methods=["POST"])
def api_regresion_lineal():
    data = request.get_json(silent=True) or {}

    try:
        n = data.get("n")
        if not isinstance(n, int) or n < 2:
            raise ValueError("n debe ser un numero entero positivo mayor o igual a 2.")

        raw_points = data.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != n:
            raise ValueError("La cantidad de puntos enviados debe coincidir con n.")

        points = []
        for index, point in enumerate(raw_points, start=1):
            if not isinstance(point, dict):
                raise ValueError(f"El punto {index} no tiene un formato valido.")

            x = parse_number(point.get("x"), f"X del punto {index}")
            y = parse_number(point.get("y"), f"Y del punto {index}")
            points.append({"x": x, "y": y})

        raw_x_test = data.get("x_test")
        x_test = None
        if raw_x_test is not None and str(raw_x_test).strip() != "":
            x_test = parse_number(raw_x_test, "x_test")

        result = calculate_linear_regression(points, x_test)
        return jsonify({"ok": True, "result": result})
    except ValueError as error:
        return jsonify({"ok": False, "message": str(error)}), 400
    except Exception:
        return jsonify({
            "ok": False,
            "message": "Ocurrio un error inesperado al calcular la regresion.",
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
