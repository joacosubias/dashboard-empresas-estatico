import yfinance as yf
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import os
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import google.generativeai as genai
from newsapi import NewsApiClient
from datetime import datetime, timedelta

# --- Configurar las APIs ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY no está configurada en los Secrets de Replit.")
    print("Por favor, ve al icono del candado en Replit y añade GEMINI_API_KEY con tu clave de Gemini.")
    exit()

try:
    newsapi = NewsApiClient(api_key=os.environ["NEWSAPI_API_KEY"])
except KeyError:
    print("Error: NEWSAPI_API_KEY no está configurada en los Secrets de Replit.")
    print("Por favor, ve al icono del candado en Replit y añade NEWSAPI_API_KEY con tu clave de NewsAPI.")
    exit()

# Asegúrate de que estas carpetas existan
if not os.path.exists('public/img'):
    os.makedirs('public/img')
if not os.path.exists('public/css'):
    os.makedirs('public/css')

# Configurar el entorno Jinja2
env = Environment(loader=FileSystemLoader('templates'))

# --- Función para formatear valores financieros ---
def format_financial_value(value):
    if value is None or pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value:,.2f}"
    else:
        return f"${value:,.2f}"

# --- Función para generar gráficos ---
def generate_chart(data, title, filename, y_label='Precio de Cierre', period='1y'):
    if data is None or data.empty:
        print(f"No hay datos para generar el gráfico: {title}")
        return False

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], color='#4CAF50')
    plt.title(title, color='#333333', fontsize=16)
    plt.xlabel('Fecha', color='#555555')
    plt.ylabel(y_label, color='#555555')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.gca().set_facecolor('#F9F9F9')
    plt.gcf().set_facecolor('#FFFFFF')

    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xticks(rotation=45, ha='right', color='#555555')
    plt.yticks(color='#555555')

    plt.tight_layout()
    plt.savefig(f'public/img/{filename}')
    plt.close()
    return True

# --- NUEVA FUNCIÓN: Obtener precio promedio en un rango de fechas ---
def get_average_price_around_date(historical_data, target_date, days_window=5):
    """
    Obtiene el precio promedio de cierre de los días hábiles
    en un rango alrededor de una fecha objetivo.
    historical_data: DataFrame de Yahoo Finance con índice de fecha y columna 'Close'.
    target_date: Objeto datetime.date o pandas.Timestamp para la fecha central.
    days_window: Número de días calendario a cada lado de la fecha objetivo
                 para formar la ventana de búsqueda.
    """
    if historical_data.empty:
        return None

    # Convertir target_date a datetime si no lo es
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()

    start_search_date = target_date - timedelta(days=days_window)
    end_search_date = target_date + timedelta(days=days_window)

    # Filtrar los datos históricos para el rango de búsqueda
    # Asegúrate de que el índice sea de tipo datetime
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        historical_data.index = pd.to_datetime(historical_data.index)

    window_data = historical_data.loc[str(start_search_date):str(end_search_date), 'Close']

    if not window_data.empty:
        return window_data.mean()
    return None


# --- Función para obtener noticias con NewsAPI y resumir con Gemini ---
def get_news_summary_with_gemini(company_name, ticker, max_links=3, current_price=None, change_1y=None, operating_income=None, net_income=None):
    model = genai.GenerativeModel('gemini-1.5-flash')

    news_links = []

    try:
        # Búsqueda de noticias usando NewsAPI
        articles = newsapi.get_everything(
            q=f'"{company_name}" OR "{ticker} stock"',
            language='en', # Noticas en inglés
            sort_by='relevancy',
            page_size=max_links * 5 # Traemos más por si filtramos algunos
        )

        # --- DEBUGGING NEWSAPI ---
        print(f"DEBUG: NewsAPI encontró {articles['totalResults']} artículos para {company_name} ({ticker}).")
        if articles['totalResults'] == 0:
            print(f"DEBUG: NewsAPI no encontró ningún artículo. No hay nada para filtrar.")
        else:
            print(f"DEBUG: Primeros 5 títulos de NewsAPI:")
            for i, article in enumerate(articles['articles'][:5]):
                print(f"  - {i+1}: {article.get('title', 'Sin título')} (Fuente: {article.get('source', {}).get('name', 'N/A')})")
        # --- FIN DEBUGGING NEWSAPI ---

        relevant_news_for_gemini_prompt = []
        count = 0
        if articles and articles['articles']:
            for article in articles['articles']:
                if count >= max_links:
                    break
                # Filtramos artículos sin título o URL
                if article.get('title') and article.get('url'):
                    relevant_news_for_gemini_prompt.append(f"Título: {article['title']}\nFuente: {article.get('source', {}).get('name', 'N/A')}\nEnlace: {article['url']}")
                    news_links.append({
                        'title': article['title'],
                        'url': article['url'],
                        'publisher': article.get('source', {}).get('name', 'N/A')
                    })
                    count += 1

        if not relevant_news_for_gemini_prompt:
            raise ValueError("NewsAPI no encontró artículos relevantes después de filtrar.")

        # Construir la información financiera para el prompt
        financial_info_for_prompt = []
        if current_price is not None:
            financial_info_for_prompt.append(f"Precio actual de la acción: ${current_price:.2f}")
        if change_1y is not None:
            financial_info_for_prompt.append(f"Cambio porcentual en el último año: {change_1y:.2f}%")
        if operating_income is not None:
            financial_info_for_prompt.append(f"Ingresos operativos más recientes: {format_financial_value(operating_income)}")
        if net_income is not None:
            financial_info_for_prompt.append(f"Ingreso neto más reciente: {format_financial_value(net_income)}")

        financial_info_text = ""
        if financial_info_for_prompt:
            financial_info_text = "\n\nInformación financiera clave:\n- " + "\n- ".join(financial_info_for_prompt)


        # Construir el prompt para Gemini (ahora en español y con datos financieros)
        prompt = (
            f"Basado en los siguientes titulares de noticias y la información financiera proporcionada sobre {company_name} ({ticker}), "
            f"genera un resumen conciso y objetivo en **español**. "
            f"El resumen debe tener entre 3 y 5 oraciones. "
            f"El resumen debe primero incluir noticias sobre expansion/reduccion del negocio o unidades del negocio, como compra o venta de activos o fusiones con otras empresas. Debe incluir si hubo algun cambio en su rating de credito. Debe incluir algun analisis de la accion y la comparacion de su precio contra en consenso. Debe cubrir las noticias más relevantes, en caso de encontrar noticias sobre deuda/solvenica incluirla en el resumen y, si la información está disponible, incluir un breve análisis del rendimiento de las acciones y/o los ingresos y/o deuda y capacidad de repago en relación con las noticias.\n"
            f"No menciones la fuente específica de las noticias en el resumen final. Si no hay noticias, o no hay información financiera, simplemente omite esa parte del análisis.\n\n"
            f"Noticias:\n"
            + "\n\n".join(relevant_news_for_gemini_prompt)
            + financial_info_text
        )

        # --- NUEVOS DEBUG PARA GEMINI ---
        print(f"DEBUG: Enviando prompt a Gemini para {company_name}. Longitud del prompt: {len(prompt)} caracteres.")
        print(f"DEBUG: Contenido del prompt (primeras 500 caracteres):\n{prompt[:500]}...")

        response = model.generate_content(prompt)
        summary = response.text
        print(f"DEBUG: Gemini generó resumen para {company_name}.")
        # --- FIN NUEVOS DEBUG PARA GEMINI ---

        return summary, news_links

    except Exception as e:
        # AQUI AGREGAMOS UN PRINT MAS ESPECIFICO PARA CAPTURAR EL ERROR COMPLETO
        print(f"ERROR FATAL AL OBTENER NOTICIAS O GENERAR RESUMEN CON GEMINI PARA {company_name}: {e}")

        # En caso de error, intentamos un último fallback con Yahoo Finance (solo si el primer intento falló)
        try:
            stock_yf = yf.Ticker(ticker)
            yf_news_items = stock_yf.news
            yf_news_for_gemini_prompt = []
            yf_news_links = []
            count_yf = 0
            for news_item in yf_news_items:
                if count_yf >= max_links:
                    break
                if news_item.get('title') and news_item.get('link'):
                    yf_news_for_gemini_prompt.append(f"Título: {news_item['title']}\nEnlace: {news_item['link']}")
                    yf_news_links.append({'title': news_item['title'], 'url': news_item['link'], 'publisher': news_item.get('publisher', 'N/A')})
                    count_yf += 1

            if yf_news_for_gemini_prompt:
                # Prompt de fallback para Yahoo Finance (también en español y con datos financieros)
                prompt_yf = (
                    f"Basado en los siguientes titulares de noticias y la información financiera proporcionada sobre {company_name} ({ticker}), "
                    f"genera un resumen conciso y objetivo en **español**. "
                    f"El resumen debe tener entre 2 y 4 oraciones. "
                    f"Debe cubrir las noticias más relevantes y, si la información está disponible, incluir un breve análisis del rendimiento de las acciones y/o los ingresos en relación con las noticias.\n"
                    f"A continuación, se listan los titulares:\n\n"
                    + "\n\n".join(yf_news_for_gemini_prompt)
                    + financial_info_text # Incluir info financiera también en el fallback
                )

                # --- NUEVOS DEBUG PARA GEMINI (FALLBACK) ---
                print(f"DEBUG: Enviando prompt de FALLBACK a Gemini para {company_name}. Longitud del prompt: {len(prompt_yf)} caracteres.")
                print(f"DEBUG: Contenido del prompt de FALLBACK (primeras 500 caracteres):\n{prompt_yf[:500]}...")
                # --- FIN NUEVOS DEBUG PARA GEMINI (FALLBACK) ---

                response_yf = model.generate_content(prompt_yf)
                summary_yf = response_yf.text
                print(f"DEBUG: Gemini generó resumen de FALLBACK para {company_name}.")
                return summary_yf, yf_news_links
            else:
                return "No se pudieron obtener noticias de la web ni de Yahoo Finance para esta empresa.", []

        except Exception as fallback_e:
            print(f"ERROR EN FALLBACK DE YAHOO FINANCE PARA {company_name}: {fallback_e}")
            return "No se pudieron obtener noticias o generar un resumen para esta empresa.", []


# --- Función principal para generar el sitio estático ---
def generate_static_site():
    print("Iniciando la generación del sitio estático...")

    # --- Título de la página ---
    page_title = "<b>Public Tenants Urbana</b>" # Aquí se define el título

    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    all_company_data = []
    all_companies_summary = []

    for ticker in tickers:
        print(f"Procesando {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Obtener historial de precios para un período más largo y completo
            # Usamos "max" para asegurar que tenemos todos los datos posibles para los cálculos
            hist = stock.history(period="max") 

            # Asegúrate de que el índice de 'hist' sea DatetimeIndex para la función get_average_price_around_date
            if not isinstance(hist.index, pd.DatetimeIndex):
                hist.index = pd.to_datetime(hist.index)


            company_name = info.get('longName', ticker)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            current_price = info.get('regularMarketPrice')

            # --- Precios históricos para variación (ahora con promedios) ---
            today = datetime.now().date()

            target_date_6m_ago = today - timedelta(days=6*30) # Aprox. 6 meses calendario
            target_date_1y_ago = today - timedelta(days=365)   # Aprox. 1 año calendario
            target_date_5y_ago = today - timedelta(days=5*365) # Aprox. 5 años calendario

            price_6m_ago = get_average_price_around_date(hist, target_date_6m_ago, days_window=7) # Ventana de 7 días (aprox. 1 semana calendario)
            price_1y_ago = get_average_price_around_date(hist, target_date_1y_ago, days_window=7)
            price_5y_ago = get_average_price_around_date(hist, target_date_5y_ago, days_window=7)


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

            # --- Datos financieros ---
            try:
                financials = stock.financials
                # Usamos .iloc[0] para acceder a la primera columna por posición, que es el dato más reciente
                # y verificamos si la Serie no está vacía
                operating_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index and not financials.loc['Operating Income'].empty else None
                net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index and not financials.loc['Net Income'].empty else None
            except Exception as e:
                print(f"Error al obtener datos financieros (Operating/Net Income) para {ticker}: {e}")
                operating_income = None
                net_income = None

            try:
                cash_flow = stock.cashflow
                # Usamos .iloc[0] para acceder a la primera columna por posición
                ebitda = cash_flow.loc['EBITDA'].iloc[0] if 'EBITDA' in cash_flow.index and not cash_flow.loc['EBITDA'].empty else None
            except Exception as e:
                print(f"Error al obtener datos de flujo de caja (EBITDA) para {ticker}: {e}")
                ebitda = None

            # --- OBTENER RESUMEN DE NOTICIAS DE LA WEB (NewsAPI + Gemini) ---
            # PASAMOS LOS DATOS FINANCIEROS Y DE PRECIOS A LA FUNCIÓN DE RESUMEN
            news_summary, news_articles_list = get_news_summary_with_gemini(
                company_name=company_name,
                ticker=ticker,
                max_links=3,
                current_price=current_price,
                change_1y=change_1y, # Usamos el cambio_1y para el análisis de Gemini
                operating_income=operating_income,
                net_income=net_income
            )

            # --- Generación de gráficos ---
            # Estos aún usan períodos directos de yfinance, lo cual es apropiado para un gráfico
            hist_1y = stock.history(period="1y")
            hist_6m = stock.history(period="6mo")
            hist_5y = stock.history(period="5y")

            generate_chart(hist_6m, f'Precio de Cierre (6 Meses) - {ticker}', f'{ticker}_6m.png', period='6mo')
            generate_chart(hist_1y, f'Precio de Cierre (1 Año) - {ticker}', f'{ticker}_1y.png', period='1y')
            generate_chart(hist_5y, f'Precio de Cierre (5 Años) - {ticker}', f'{ticker}_5y.png', period='5y')

            # Generación de gráfico financiero (si hay datos disponibles)
            if not financials.empty:
                # Verificamos si las columnas existen antes de intentar graficarlas
                columns_to_plot = []
                if 'Operating Income' in financials.index:
                    columns_to_plot.append('Operating Income')
                if 'Net Income' in financials.index:
                    columns_to_plot.append('Net Income')

                if columns_to_plot:
                    financial_data_for_plot = financials.loc[columns_to_plot].transpose().iloc[:4]
                    if not financial_data_for_plot.empty:
                        plt.figure(figsize=(10, 6))
                        financial_data_for_plot.plot(kind='bar', ax=plt.gca(), color=['#66BB6A', '#FFA726'][:len(columns_to_plot)])
                        plt.title(f'Ingresos y Beneficios Netos - {ticker}', color='#333333', fontsize=16)
                        plt.xlabel('Año', color='#555555')
                        plt.ylabel('Valor ($)', color='#555555')

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
                    print(f"No se encontraron 'Operating Income' o 'Net Income' para graficar para {ticker}.")
            else:
                print(f"No hay datos financieros para generar el gráfico: {ticker}")


            # --- Determinar el color del semáforo para la tabla ---
            semaphore_color = "gray"
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
                'news_summary': news_summary,
                'news_articles': news_articles_list
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
            # Captura cualquier error en el procesamiento de un ticker y registra en la consola
            print(f"ERROR GENERAL PROCESANDO {ticker}: {e}")
            # Asegura que el ticker siempre aparezca en la tabla de resumen y en la sección de detalles
            all_company_data.append({
                'ticker': ticker,
                'name': f"{ticker} (Error)",
                'sector': 'N/A', 'industry': 'N/A',
                'current_price': 'N/A', 'change_6m': 'N/A',
                'change_1y': 'N/A', 'change_5y': 'N/A',
                'operating_income': 'N/A', 'net_income': 'N/A', 'ebitda': 'N/A',
                'news_summary': f'No se pudieron obtener datos completos o un resumen de noticias para esta empresa debido a un error: {e}',
                'news_articles': []
            })
            all_companies_summary.append({
                'ticker': ticker,
                'name': f"{ticker} (Error)",
                'current_price': 'N/A', 'change_6m': 'N/A',
                'change_1y': 'N/A', 'change_5y': 'N/A',
                'semaphore_color': 'gray'
            })


    # --- Generar el archivo CSS principal (sin cambios aquí) ---
    css_content = """
    body {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        margin: 0;
        padding: 20px;
        background-color: #F8F8F8;
        color: #333333;
        line-height: 1.6;
    }

    .container {
        max-width: 1200px;
        margin: 20px auto;
        padding: 20px;
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    h1, h2, h3 {
        color: #222222;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 300;
    }

    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 40px;
        font-size: 0.9em;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .summary-table th, .summary-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #EEEEEE;
    }

    .summary-table th {
        background-color: #F2F2F2;
        color: #555555;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .summary-table tbody tr:hover {
        background-color: #FDFDFD;
    }

    .summary-table .change-positive {
        color: #28a745;
        font-weight: bold;
    }

    .summary-table .change-negative {
        color: #dc3545;
        font-weight: bold;
    }

    .summary-table .change-neutral {
        color: #6c757d;
        font-weight: bold;
    }

    .semaphore-red {
        background-color: #ffe0e0;
        border-left: 5px solid #dc3545;
    }
    .semaphore-yellow {
        background-color: #fff9e0;
        border-left: 5px solid #ffc107;
    }
    .semaphore-green {
        background-color: #e0ffe0;
        border-left: 5px solid #28a745;
    }
    .semaphore-gray {
        background-color: #f0f0f0;
        border-left: 5px solid #cccccc;
    }

    .company-section {
        background-color: #FFFFFF;
        padding: 30px;
        margin-bottom: 30px;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-top: 5px solid #6c757d;
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
        color: #007BFF;
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
        all_companies_summary=all_companies_summary,
        page_title=page_title # Pasamos el título a la plantilla
    )

    with open('public/index.html', 'w') as f:
        f.write(output)
    print("Sitio estático generado en la carpeta 'public/'")
    print("Ahora puedes subir el contenido de la carpeta 'public' a Netlify o Vercel.")

# Ejecutar la función principal
if __name__ == '__main__':
    generate_static_site()