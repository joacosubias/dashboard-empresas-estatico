import yfinance as yf
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import os
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

# Asegúrate de que esta carpeta exista
if not os.path.exists('public/img'):
    os.makedirs('public/img')

# Asegúrate de que esta carpeta exista
if not os.path.exists('public/css'):
    os.makedirs('public/css')

# Configurar el entorno Jinja2
env = Environment(loader=FileSystemLoader('templates'))

# --- Función para obtener resumen de Wikipedia ---
def get_wikipedia_summary(company_name):
    try:
        # Intenta primero con la API de Wikipedia en español
        url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{company_name.replace(' ', '_')}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if 'extract' in data:
            return data['extract']

        # Si no se encuentra en español, intenta con la API en inglés
        url_en = f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name.replace(' ', '_')}"
        response_en = requests.get(url_en, timeout=5)
        response_en.raise_for_status()
        data_en = response_en.json()
        if 'extract' in data_en:
            return data_en['extract']

        return "Resumen no disponible."
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener resumen de Wikipedia para {company_name}: {e}")
        return "Resumen no disponible."

# --- Función para formatear valores financieros ---
def format_financial_value(value):
    if value is None or pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:,.2f}K"
    else:
        return f"${value:,.2f}"

# --- Función para generar gráficos ---
def generate_chart(data, title, filename, y_label='Precio de Cierre', period='1y'):
    if data is None or data.empty:
        print(f"No hay datos para generar el gráfico: {title}")
        return False

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], color='#4CAF50') # Color verde para la línea
    plt.title(title, color='#333333', fontsize=16) # Título en gris oscuro
    plt.xlabel('Fecha', color='#555555') # Etiquetas en gris medio
    plt.ylabel(y_label, color='#555555')
    plt.grid(True, linestyle='--', alpha=0.7) # Cuadrícula sutil

    plt.gca().set_facecolor('#F9F9F9') # Fondo del área del gráfico en gris claro
    plt.gcf().set_facecolor('#FFFFFF') # Fondo de la figura en blanco

    # Formatear el eje Y como moneda (ej. $100, $200)
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    # Rotar las etiquetas del eje X si hay muchas
    plt.xticks(rotation=45, ha='right', color='#555555') # Etiquetas en gris medio
    plt.yticks(color='#555555') # Etiquetas en gris medio

    plt.tight_layout()
    plt.savefig(f'public/img/{filename}')
    plt.close()
    return True

# --- Función principal para generar el sitio estático ---
def generate_static_site():
    print("Iniciando la generación del sitio estático...")

    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    all_company_data = []
    all_companies_summary = [] # Lista para los datos de la tabla resumen

    for ticker in tickers:
        print(f"Procesando {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5y")

            # --- Datos de la empresa ---
            company_name = info.get('longName', ticker)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            current_price = info.get('regularMarketPrice')

            # --- Resumen de Wikipedia ---
            wikipedia_summary = get_wikipedia_summary(company_name)

            # --- Precios históricos para variación ---
            price_6m_ago = hist['Close'].iloc[-126] if len(hist) >= 126 else None # Aprox 6 meses (126 días hábiles)
            price_1y_ago = hist['Close'].iloc[-252] if len(hist) >= 252 else None # Aprox 1 año (252 días hábiles)
            price_5y_ago = hist['Close'].iloc[0] if len(hist) >= 1260 else None # Aprox 5 años (1260 días hábiles)

            # --- Variaciones porcentuales ---
            change_6m = None
            if current_price and price_6m_ago is not None and price_6m_ago != 0:
                change_6m = ((current_price - price_6m_ago) / price_6m_ago) * 100

            change_1y = None
            if current_price and price_1y_ago is not None and price_1y_ago != 0:
                change_1y = ((current_price - price_1y_ago) / price_1y_ago) * 100

            change_5y = None
            if current_price and price_5y_ago is not None and price_5y_ago != 0:
                change_5y = ((current_price - price_5y_ago) / price_5y_ago) * 100

            # --- Datos financieros (Balance Sheet / Income Statement) ---
            # Asegúrate de que los datos financieros se obtengan correctamente
            # Aquí un ejemplo de cómo obtenerlos si no los tienes
            try:
                financials = stock.financials
                operating_income = financials.loc['Operating Income'][0] if 'Operating Income' in financials.index else None
                net_income = financials.loc['Net Income'][0] if 'Net Income' in financials.index else None
            except Exception as e:
                print(f"Error al obtener datos financieros para {ticker}: {e}")
                operating_income = None
                net_income = None

            try:
                cash_flow = stock.cashflow
                ebitda = cash_flow.loc['EBITDA'][0] if 'EBITDA' in cash_flow.index else None
            except Exception as e:
                print(f"Error al obtener datos de flujo de caja para {ticker}: {e}")
                ebitda = None

            # --- Generación de gráficos ---
            # Ajustar los periodos para que los gráficos tengan sentido si no hay 5y de historia
            hist_1y = stock.history(period="1y")
            hist_6m = stock.history(period="6mo")
            hist_5y = stock.history(period="5y") # Mantener para la variación, aunque el gráfico sea solo de 1y

            generate_chart(hist_6m, f'Precio de Cierre (6 Meses) - {ticker}', f'{ticker}_6m.png', period='6mo')
            generate_chart(hist_1y, f'Precio de Cierre (1 Año) - {ticker}', f'{ticker}_1y.png', period='1y')
            generate_chart(hist_5y, f'Precio de Cierre (5 Años) - {ticker}', f'{ticker}_5y.png', period='5y')

            # --- Datos para el gráfico financiero ---
            # Obtener datos de ingresos y beneficio para el gráfico financiero
            if not financials.empty:
                # Usar transpose() para que las fechas sean el índice si no lo son
                # Y seleccionar las últimas 4 columnas (años) para el gráfico
                financial_data_for_plot = financials.loc[['Operating Income', 'Net Income']].transpose().iloc[:4]
                if not financial_data_for_plot.empty:
                    plt.figure(figsize=(10, 6))
                    financial_data_for_plot.plot(kind='bar', ax=plt.gca(), color=['#66BB6A', '#FFA726']) # Colores amigables
                    plt.title(f'Ingresos y Beneficios Netos - {ticker}', color='#333333', fontsize=16)
                    plt.xlabel('Año', color='#555555')
                    plt.ylabel('Valor ($)', color='#555555')

                    # Formatear el eje Y como moneda en millones o billones
                    formatter = FuncFormatter(lambda x, p: format_financial_value(x))
                    plt.gca().yaxis.set_major_formatter(formatter)

                    plt.xticks(rotation=45, ha='right', color='#555555')
                    plt.yticks(color='#555555')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.gca().set_facecolor('#F9F9F9')
                    plt.gcf().set_facecolor('#FFFFFF')
                    plt.tight_layout()
                    plt.savefig(f'public/img/{ticker}_financial.png')
                    plt.close()
                else:
                    print(f"No hay datos financieros suficientes para el gráfico: {ticker}")
            else:
                print(f"No hay datos financieros para generar el gráfico: {ticker}")

            # --- Determinar el color del semáforo para la tabla ---
            # Basado en la variación de 1 año. Ajusta los umbrales si lo deseas.
            semaphore_color = "gray"  # Default
            if change_1y is not None:
                if change_1y >= 20:
                    semaphore_color = "green"
                elif change_1y >= 0:
                    semaphore_color = "yellow"
                else:
                    semaphore_color = "red"

            # --- Almacenar todos los datos de la empresa para la plantilla ---
            all_company_data.append({
                'ticker': ticker,
                'name': company_name,
                'sector': sector,
                'industry': industry,
                'current_price': f"${current_price:.2f}" if current_price else "N/A",
                'change_6m': f"{change_6m:.2f}%" if change_6m is not None else "N/A",
                'change_1y': f"{change_1y:.2f}%" if change_1y is not None else "N/A",
                'change_5y': f"{change_5y:.2f}%" if change_5y is not None else "N/A",
                'operating_income': format_financial_value(operating_income),
                'net_income': format_financial_value(net_income),
                'ebitda': format_financial_value(ebitda),
                'wikipedia_summary': wikipedia_summary,
            })

            # --- Datos específicos para la tabla resumen ---
            all_companies_summary.append({
                'ticker': ticker,
                'name': company_name,
                'current_price': f"${current_price:.2f}" if current_price else "N/A",
                'change_6m': f"{change_6m:.2f}%" if change_6m is not None else "N/A",
                'change_1y': f"{change_1y:.2f}%" if change_1y is not None else "N/A",
                'change_5y': f"{change_5y:.2f}%" if change_5y is not None else "N/A",
                'semaphore_color': semaphore_color
            })

        except Exception as e:
            print(f"Error procesando {ticker}: {e}")
            # Añadir datos básicos con "N/A" si hay un error para que la tabla no se rompa
            all_company_data.append({
                'ticker': ticker,
                'name': f"{ticker} (Error)",
                'sector': 'N/A', 'industry': 'N/A',
                'current_price': 'N/A', 'change_6m': 'N/A',
                'change_1y': 'N/A', 'change_5y': 'N/A',
                'operating_income': 'N/A', 'net_income': 'N/A', 'ebitda': 'N/A',
                'wikipedia_summary': 'No se pudieron obtener datos para esta empresa debido a un error.',
            })
            all_companies_summary.append({
                'ticker': ticker,
                'name': f"{ticker} (Error)",
                'current_price': 'N/A', 'change_6m': 'N/A',
                'change_1y': 'N/A', 'change_5y': 'N/A',
                'semaphore_color': 'gray'
            })


    # --- Generar el archivo CSS principal ---
    css_content = """
    body {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        margin: 0;
        padding: 20px;
        background-color: #F8F8F8; /* Gris muy claro */
        color: #333333; /* Gris oscuro para texto principal */
        line-height: 1.6;
    }

    .container {
        max-width: 1200px;
        margin: 20px auto;
        padding: 20px;
        background-color: #FFFFFF; /* Fondo blanco */
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); /* Sombra sutil */
    }

    h1, h2, h3 {
        color: #222222; /* Gris casi negro para títulos */
        text-align: center;
        margin-bottom: 25px;
        font-weight: 300; /* Fuente más ligera */
    }

    /* Estilos de la tabla resumen */
    .summary-table {
        width: 100%;
        border-collapse: collapse; /* Eliminar bordes dobles */
        margin-bottom: 40px;
        font-size: 0.9em;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .summary-table th, .summary-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #EEEEEE; /* Líneas divisorias suaves */
    }

    .summary-table th {
        background-color: #F2F2F2; /* Gris claro para encabezados */
        color: #555555; /* Gris medio para texto de encabezado */
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .summary-table tbody tr:hover {
        background-color: #FDFDFD; /* Ligero resaltado al pasar el ratón */
    }

    .summary-table .change-positive {
        color: #28a745; /* Verde para positivo */
        font-weight: bold;
    }

    .summary-table .change-negative {
        color: #dc3545; /* Rojo para negativo */
        font-weight: bold;
    }

    .summary-table .change-neutral {
        color: #6c757d; /* Gris para neutral o sin cambio */
    }

    /* Colores del semáforo */
    .semaphore-red {
        background-color: #ffe0e0; /* Rojo claro */
        border-left: 5px solid #dc3545; /* Barra lateral roja */
    }
    .semaphore-yellow {
        background-color: #fff9e0; /* Amarillo claro */
        border-left: 5px solid #ffc107; /* Barra lateral amarilla */
    }
    .semaphore-green {
        background-color: #e0ffe0; /* Verde claro */
        border-left: 5px solid #28a745; /* Barra lateral verde */
    }
    .semaphore-gray {
        background-color: #f0f0f0; /* Gris muy claro */
        border-left: 5px solid #cccccc; /* Barra lateral gris */
    }

    /* Estilos para las secciones de empresas individuales */
    .company-section {
        background-color: #FFFFFF;
        padding: 30px;
        margin-bottom: 30px;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-top: 5px solid #6c757d; /* Borde superior gris */
    }

    .company-section h3 {
        text-align: left;
        margin-top: 0;
        margin-bottom: 15px;
        color: #333333;
    }

    .company-info p {
        margin: 5px 0;
        color: #555555;
    }

    .company-info strong {
        color: #222222;
    }

    .charts-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 25px;
    }

    .chart-item img {
        max-width: 100%;
        height: auto;
        display: block;
        border: 1px solid #EEEEEE;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .news-section, .wikipedia-summary-section {
        background-color: #FDFDFD;
        padding: 20px;
        margin-top: 20px;
        border-radius: 6px;
        border: 1px solid #EEEEEE;
    }

    .news-section h4, .wikipedia-summary-section h4 {
        color: #444444;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .news-item {
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 1px dashed #E0E0E0;
    }

    .news-item:last-child {
        border-bottom: none;
    }

    .news-item a {
        color: #007BFF; /* Azul para enlaces */
        text-decoration: none;
        font-weight: 500;
    }

    .news-item a:hover {
        text-decoration: underline;
    }

    .news-item p {
        font-size: 0.9em;
        color: #666666;
    }

    .news-item span.source {
        font-size: 0.8em;
        color: #999999;
    }

    footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #EEEEEE;
        color: #777777;
        font-size: 0.9em;
    }
    """
    with open('public/css/style.css', 'w') as f:
        f.write(css_content)
    print("Archivo CSS generado en public/css/style.css")


    # --- Renderizar la plantilla HTML ---
    template = env.get_template('index.html')
    output = template.render(
        companies=all_company_data,
        all_companies_summary=all_companies_summary
    )

    with open('public/index.html', 'w') as f:
        f.write(output)
    print("Sitio estático generado en la carpeta 'public/'")
    print("Ahora puedes subir el contenido de la carpeta 'public' a Netlify o Vercel.")

# Ejecutar la función principal
if __name__ == '__main__':
    generate_static_site()