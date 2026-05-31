# Aplicación de Regresión con Flask

Aplicación web en Python que permite calcular regresión lineal, cuadrática y exponencial, mostrar la ecuación de ajuste, probar valores y graficar los datos.

## Archivos principales

- `app.py`: aplicación Flask y lógica de regresión.
- `templates/index.html`: formulario de entrada y rúbrica.
- `templates/result.html`: resultado con ecuación y gráfico.
- `requirements.txt`: dependencias necesarias.

## Cómo ejecutar

1. Abre terminal en `c:\Users\jonat\Desktop\regresion`.
2. Activa el entorno virtual:
   - ` .venv\Scripts\activate`
3. Instala dependencias:
   - `pip install -r requirements.txt`
4. Ejecuta la aplicación:
   - `python app.py`
5. Abre el navegador en:
   - `http://127.0.0.1:5000`

## Uso

- Ingresa puntos separados por linea en formato `x, y`.
- Selecciona el tipo de regresión.
- Opcional: prueba un valor `x` para obtener `y` estimado.
- La aplicación mostrará el gráfico con los puntos originales, la curva de ajuste y el punto probado.
