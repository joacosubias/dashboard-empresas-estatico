import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from newsapi import NewsApiClient
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta # Asegúrate de que datetime está importado
import requests
from bs4 import BeautifulSoup
import matplotlib.ticker as mticker

# Configuración de la aplicación Flask
app = Flask(__name__)

# --- Configuración de API Keys ---
# Asegúrate de que NEWS_API_KEY esté configurada en los Secrets de Replit
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
if not NEWS_API_KEY:
    print("Advertencia: NEWS_API_KEY no configurada. La funcionalidad de noticias no estará disponible.")
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None

# Lista de tickers para el dashboard (puedes cargarla desde tickers.txt)
def load_tickers(filename="tickers.txt"):
    try:
        with open(filename, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"Error: El archivo {filename} no fue encontrado.")
        return []

TICKERS = load_tickers()

# --- Funciones de Utilidad para Gráficos ---

def get_stock_data(ticker_symbol, period="6mo"):
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period=period)
        if history.empty:
            return None
        # Asegurarse de que el índice sea DatetimeIndex
        history.index = pd.to_datetime(history.index)
        return history
    except Exception as e:
        print(f"Error al obtener datos de {ticker_symbol} para {period}: {e}")
        return None

def get_financial_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        financials = ticker.financials
        if financials.empty:
            return None, None

        # Intentar obtener 'Operating Income' y 'Total Revenue'
        operating_income_row = financials.loc['Operating Income'] if 'Operating Income' in financials.index else None
        total_revenue_row = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None

        # Convertir a DataFrame y transponer para tener años como índice
        op_income_df = pd.DataFrame(operating_income_row).T if operating_income_row is not None else pd.DataFrame()
        revenue_df = pd.DataFrame(total_revenue_row).T if total_revenue_row is not None else pd.DataFrame()

        # Asegurarse de que el índice sea DatetimeIndex
        if not op_income_df.empty:
            # Reorganizar el DataFrame para que el índice sea la fecha y la columna sea el valor
            op_income_df = op_income_df.T
            op_income_df.index = pd.to_datetime(op_income_df.index)
            # Asegurarse de que la columna se llame 'Operating Income'
            if 'Operating Income' not in op_income_df.columns:
                 # Si no está, asumimos que es la única columna y la renombramos
                op_income_df.columns = ['Operating Income']
        if not revenue_df.empty:
            # Reorganizar el DataFrame para que el índice sea la fecha y la columna sea el valor
            revenue_df = revenue_df.T
            revenue_df.index = pd.to_datetime(revenue_df.index)
            # Asegurarse de que la columna se llame 'Total Revenue'
            if 'Total Revenue' not in revenue_df.columns:
                 # Si no está, asumimos que es la única columna y la renombramos
                revenue_df.columns = ['Total Revenue']

        return op_income_df, revenue_df

    except Exception as e:
        print(f"Error al obtener datos financieros de {ticker_symbol}: {e}")
        return None, None

def calculate_semaphor_color(metrics):
    # Aquí puedes definir tus reglas para el semáforo
    # Esto es solo un ejemplo simplificado
    price = metrics.get('currentPrice')
    market_cap = metrics.get('marketCap')
    pe_ratio = metrics.get('trailingPE')
    dividend_yield = metrics.get('dividendYield')

    score = 0
    if price and price > 0: score += 1
    if market_cap and market_cap > 1e9: score += 1 # Ejemplo: capitalización > 1 billón USD
    if pe_ratio and pe_ratio is not None and 0 < pe_ratio < 25: score += 1 # Ejemplo: PE razonable
    if dividend_yield is not None and dividend_yield > 0.01: score += 1 # Ejemplo: dividendos > 1%

    if score >= 3:
        return "green"
    elif score >= 2:
        return "yellow"
    else:
        return "red"

def generate_chart_image(history_data, title, y_label, file_path):
    if history_data is None or history_data.empty:
        print(f"No hay datos para generar el gráfico: {title}")
        return None

    plt.style.use('dark_background') # Estilo oscuro para el gráfico
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel(y_label, color=color)
    ax1.plot(history_data.index, history_data['Close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Formato de fecha en el eje X (mejorado para Matplotlib)
    plt.xticks(rotation=45, ha='right') # Rotar etiquetas y alinear a la derecha

    # Formato de moneda para el eje Y
    formatter = mticker.FormatStrFormatter('$%.2f')
    ax1.yaxis.set_major_formatter(formatter)

    plt.title(title)
    fig.tight_layout()

    # Guardar el gráfico como archivo PNG
    plt.savefig(file_path, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig) # Cierra la figura para liberar memoria
    return file_path

def generate_financial_chart(op_income_df, revenue_df, ticker_symbol, file_path):
    plt.style.use('dark_background') # Estilo oscuro
    fig, ax = plt.subplots(figsize=(10, 6))

    # Asegurarse de que los DataFrames no estén vacíos antes de intentar acceder a ellos
    if not revenue_df.empty and 'Total Revenue' in revenue_df.columns:
        # Usar el índice (fechas) y la columna 'Total Revenue'
        ax.plot(revenue_df.index, revenue_df['Total Revenue'], label='Total Revenue', color='tab:green', marker='o')
    else:
        print(f"Advertencia: 'Total Revenue' no encontrado o vacío para {ticker_symbol}")

    if not op_income_df.empty and 'Operating Income' in op_income_df.columns:
        # Usar el índice (fechas) y la columna 'Operating Income'
        ax.plot(op_income_df.index, op_income_df['Operating Income'], label='Operating Income', color='tab:red', marker='x')
    else:
        print(f"Advertencia: 'Operating Income' no encontrado o vacío para {ticker_symbol}")

    ax.set_xlabel('Año')
    ax.set_ylabel('Millones de USD')
    ax.set_title(f'Ingresos y Ganancias Operativas de {ticker_symbol}')
    ax.legend()
    plt.xticks(rotation=45, ha='right') # Rotar etiquetas y alinear a la derecha
    fig.tight_layout()

    # Guardar el gráfico como archivo PNG
    plt.savefig(file_path, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig) # Cierra la figura para liberar memoria
    return file_path

def get_company_news(company_name):
    articles = []
    if newsapi:
        try:
            # Buscar noticias por el nombre de la empresa
            response = newsapi.get_everything(
                q=company_name,
                language='es', # O inglés si prefieres noticias en inglés
                sort_by='relevancy',
                page_size=5 # Número de noticias
            )
            if response['status'] == 'ok':
                articles = response['articles']
            else:
                print(f"Error al obtener noticias: {response.get('message', 'Mensaje desconocido')}")
        except Exception as e:
            print(f"Error en la llamada a NewsAPI: {e}")
    return articles

def get_wikipedia_summary(company_name):
    try:
        # Usa el endpoint de Wikipedia API para obtener el resumen
        # Intentamos buscar con el nombre de la empresa tal cual, pero la API es sensible.
        # Para evitar el 404, si el nombre es muy largo o tiene caracteres especiales, podríamos
        # intentar buscar solo la primera parte o simplificar.
        safe_company_name = company_name.replace(' ', '_').replace('&', '%26').replace('.', '') # Eliminar '.' y codificar '&'
        response = requests.get(
            f"https://es.wikipedia.org/api/rest_v1/page/summary/{safe_company_name}"
        )
        response.raise_for_status()  # Lanza una excepción para errores HTTP (como 404)
        data = response.json()
        return data.get('extract', 'Resumen no disponible.')
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener resumen de Wikipedia para {company_name}: {e}")
        return "Resumen no disponible."


# --- Función principal para generar el sitio estático ---

def generate_static_site():
    print("Iniciando la generación del sitio estático...")

    # Asegurarse de que las carpetas de salida existan
    output_dir = 'public'
    img_dir = os.path.join(output_dir, 'img')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    companies_data = []

    for ticker_symbol in TICKERS:
        print(f"Procesando {ticker_symbol}...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            # Datos para el resumen general
            company_name = info.get('longName', ticker_symbol)
            current_price = info.get('currentPrice', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            dividend_yield = info.get('dividendYield', 'N/A')

            # Calcular el color del semáforo
            semaphor_color = calculate_semaphor_color(info)

            # Generar gráficos y obtener sus rutas de archivo
            # Las URLs para el HTML deben ser relativas a la carpeta 'public'
            chart_6m_url = f'img/{ticker_symbol}_6m.png'
            chart_1y_url = f'img/{ticker_symbol}_1y.png'
            chart_5y_url = f'img/{ticker_symbol}_5y.png'
            financial_chart_url = f'img/{ticker_symbol}_financial.png'

            # Las rutas para guardar los archivos en el sistema de archivos
            chart_6m_path = os.path.join(img_dir, f'{ticker_symbol}_6m.png')
            chart_1y_path = os.path.join(img_dir, f'{ticker_symbol}_1y.png')
            chart_5y_path = os.path.join(img_dir, f'{ticker_symbol}_5y.png')
            financial_chart_path = os.path.join(img_dir, f'{ticker_symbol}_financial.png')

            history_6m = get_stock_data(ticker_symbol, "6mo")
            generate_chart_image(history_6m, f'Precio de Cierre (6 Meses) - {ticker_symbol}', 'Precio (USD)', chart_6m_path)

            history_1y = get_stock_data(ticker_symbol, "1y")
            generate_chart_image(history_1y, f'Precio de Cierre (1 Año) - {ticker_symbol}', 'Precio (USD)', chart_1y_path)

            history_5y = get_stock_data(ticker_symbol, "5y")
            generate_chart_image(history_5y, f'Precio de Cierre (5 Años) - {ticker_symbol}', 'Precio (USD)', chart_5y_path)

            op_income_df, revenue_df = get_financial_data(ticker_symbol)
            generate_financial_chart(op_income_df, revenue_df, ticker_symbol, financial_chart_path)

            # Obtener noticias y resumen de Wikipedia
            news_articles = get_company_news(company_name)
            wikipedia_summary = get_wikipedia_summary(company_name)

            companies_data.append({
                'ticker': ticker_symbol,
                'name': company_name,
                'current_price': current_price,
                'pe_ratio': pe_ratio,
                'market_cap': market_cap,
                'dividend_yield': f"{dividend_yield*100:.2f}%" if isinstance(dividend_yield, (int, float)) else 'N/A',
                'semaphor_color': semaphor_color,
                'chart_6m_url': chart_6m_url, # Usa la URL relativa
                'chart_1y_url': chart_1y_url,
                'chart_5y_url': chart_5y_url,
                'financial_chart_url': financial_chart_url,
                'news_articles': news_articles,
                'wikipedia_summary': wikipedia_summary
            })

        except Exception as e:
            print(f"Error procesando {ticker_symbol}: {e}")
            companies_data.append({
                'ticker': ticker_symbol,
                'name': ticker_symbol,
                'current_price': 'N/A',
                'pe_ratio': 'N/A',
                'market_cap': 'N/A',
                'dividend_yield': 'N/A',
                'semaphor_color': 'red', # Rojo si hay error
                'chart_6m_url': '',
                'chart_1y_url': '',
                'chart_5y_url': '',
                'financial_chart_url': '',
                'news_articles': [],
                'wikipedia_summary': 'Error al cargar datos.'
            })

    # Renderizar la plantilla HTML con todos los datos
    # Obtener la fecha y hora actual para el footer
    current_generation_time = datetime.now().strftime("%d/%m/%Y %H:%M") # Formato deseado

    rendered_html = render_template('index.html',
                                    companies=companies_data,
                                    generation_time=current_generation_time) # Pasa la variable a la plantilla

    # Guardar el HTML renderizado en la carpeta 'public'
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    print(f"Sitio estático generado en la carpeta '{output_dir}/'")
    print("Ahora puedes subir el contenido de la carpeta 'public' a Netlify o Vercel.")

# --- Ejecución del generador estático ---
if __name__ == '__main__':
    # Crear un contexto de aplicación manualmente para render_template
    with app.app_context():
        generate_static_site()

    # No necesitamos app.run() aquí si solo vamos a generar estáticos