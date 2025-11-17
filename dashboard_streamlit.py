"""
Dashboard Interativo Streamlit - Análise Térmica 3TC
Visualização de dados de temperatura e correlações
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import re
from datetime import datetime, timedelta
from functools import lru_cache
from scipy import stats
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import requests
import warnings
warnings.filterwarnings('ignore')

# Diretórios fixos para os cenários pré e pós instalação
PRE_INSTALL_DIR = Path("Dados de entrada pré instalação")
POST_INSTALL_DIR = Path("Dados de entrada pós instalação") / "sensores_pos"

# Configuração default da cidade (Cabedelo/PB)
DEFAULT_LOCATION = {
    "name": "Cabedelo",
    "admin1": "Paraíba",
    "country_code": "BR",
    "latitude": -6.9711,
    "longitude": -34.8378,
    "timezone": "America/Fortaleza"
}

# Configuração da página
st.set_page_config(
    page_title="Análise Térmica 3TC",
    page_icon="🌡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNÃ‡Ã•ES AUXILIARES (mesmas do script principal)
# ============================================================================

def sniff_delim_and_encoding(filepath: Path):
    """Detecta delimitador e encoding do CSV"""
    with open(filepath, 'rb') as f:
        sample = f.read(4096)
        # Tenta encodings brasileiros primeiro (mais comum)
        for enc in ["latin-1", "cp1252", "iso-8859-1", "utf-8"]:
            try:
                text = sample.decode(enc, errors="ignore")
                semi = text.count(";")
                comma = text.count(",")
                return (";" if semi > comma else ","), enc
            except (UnicodeDecodeError, Exception):
                continue
    # Fallback seguro
    return ";", "latin-1"

def parse_sensor_id(filename: str) -> str:
    """Extrai ID do sensor do nome do arquivo"""
    m = re.search(r"(EL\d+)", filename)
    return m.group(1) if m else Path(filename).stem

def parse_timestamp_series(raw_series: pd.Series) -> pd.Series:
    """Parse de timestamps com múltiplos formatos"""
    # Ordem de tentativa: formato ISO primeiro (mais comum nos CSVs dos sensores)
    # depois formatos brasileiros como fallback
    fmts = [
        "%Y-%m-%d %H:%M:%S",  # 2024-02-15 18:00:00 (FORMATO ISO - MAIS COMUM NOS CSVs)
        "%Y-%m-%d %H:%M",     # 2024-02-15 18:00
        "%d/%m/%Y %H:%M:%S",  # 15/02/2024 18:00:00 (formato brasileiro)
        "%d/%m/%Y %H:%M",     # 15/02/2024 18:00 (formato brasileiro)
        "%Y/%m/%d %H:%M:%S",  # 2024/02/15 18:00:00
        "%Y/%m/%d %H:%M",     # 2024/02/15 18:00
    ]
    
    for fmt in fmts:
        try:
            # Para formato ISO não precisa dayfirst, para formato brasileiro sim
            dayfirst = fmt.startswith("%d/")
            ts = pd.to_datetime(raw_series, format=fmt, errors="coerce", utc=False, dayfirst=dayfirst)
            # Verifica se pelo menos 90% dos valores foram parseados corretamente
            if ts.notna().sum() > len(raw_series) * 0.9:
                return ts
        except Exception:
            continue
    
    # Fallback: tenta inferir automaticamente (tenta ambos os formatos)
    try:
        # Primeiro tenta sem dayfirst (formato ISO)
        ts = pd.to_datetime(raw_series, errors="coerce", utc=False, dayfirst=False, infer_datetime_format=False)
        if ts.notna().sum() > len(raw_series) * 0.9:
            return ts
    except Exception:
        pass
    
    try:
        # Depois tenta com dayfirst (formato brasileiro)
        ts = pd.to_datetime(raw_series, errors="coerce", utc=False, dayfirst=True, infer_datetime_format=False)
        if ts.notna().sum() > len(raw_series) * 0.9:
            return ts
    except Exception:
        pass
    
    # Último recurso: retorna como está e deixa o pandas lidar (sem dayfirst por padrão)
    return pd.to_datetime(raw_series, errors="coerce", utc=False, dayfirst=False)


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@lru_cache(maxsize=128)
def geocode_city(query: str, count: int = 5, country_code: str = None) -> List[Dict]:
    """Consulta a API de geocodificação da Open-Meteo e retorna possíveis cidades."""
    if not query or len(query.strip()) < 3:
        return []
    base = query.strip()
    search_terms = [base]
    if "," in base:
        primary = base.split(",", 1)[0].strip()
        if primary and primary not in search_terms:
            search_terms.append(primary)
    if " " in base:
        first_word = base.split(" ", 1)[0].strip()
        if first_word and first_word not in search_terms:
            search_terms.append(first_word)
    for term in search_terms:
        if len(term) < 3:
            continue
        params = {
            "name": term,
            "count": count * 2 if country_code else count,  # Busca mais se filtrar por país
            "language": "pt",
            "format": "json"
        }
        # Adiciona filtro por país se especificado
        if country_code:
            params["country_codes"] = country_code
        
        try:
            response = requests.get(OPEN_METEO_GEOCODING_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if results:
                # Filtra apenas cidades do Brasil se country_code for BR
                if country_code:
                    results = [r for r in results if r.get('country_code', '').upper() == country_code.upper()]
                if results:
                    return results[:count]  # Retorna apenas o número solicitado
        except Exception:
            continue
    return []


def fetch_external_temperature_from_api(
    latitude: float,
    longitude: float,
    start_ts: datetime,
    end_ts: datetime,
    tz: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Baixa temperatura externa horária via Open-Meteo para o período informado."""
    if start_ts is None or end_ts is None:
        return None
    
    try:
        start = pd.to_datetime(start_ts).tz_localize(None)
        end = pd.to_datetime(end_ts).tz_localize(None)
        if start > end:
            start, end = end, start
        
        # Verifica se as datas são muito antigas (Open-Meteo Archive tem limitações)
        # A API geralmente tem dados de 1940 até alguns dias atrás
        hoje = pd.Timestamp.now()
        if end > hoje:
            end = hoje
        if start > hoje:
            return None
        
        params = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": "temperature_2m",
            "timezone": tz or "auto",
            "temperature_unit": "celsius"
        }
        
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Verifica se há erros na resposta
        if "error" in data:
            raise ValueError(f"API retornou erro: {data.get('reason', 'Erro desconhecido')}")
        
        hourly = data.get("hourly") or {}
        times = hourly.get("time")
        temps = hourly.get("temperature_2m")
        
        if not times or temps is None:
            # Tenta verificar se há mensagem de erro na resposta
            error_msg = data.get("error", {}).get("reason", "Resposta da API sem série de temperatura horária.")
            raise ValueError(f"API não retornou dados: {error_msg}")
        
        if len(times) == 0 or len(temps) == 0:
            raise ValueError("API retornou lista vazia de dados de temperatura.")
        
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(times, errors="coerce"),
                "temp_externa": temps
            }
        )
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        
        if df.empty:
            return None
        
        df.index = df.index.tz_localize(None)
        return df
        
    except requests.exceptions.Timeout:
        raise Exception("Timeout ao consultar API Open-Meteo. Tente novamente.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erro de conexão com API Open-Meteo: {str(e)}")
    except ValueError as e:
        raise
    except Exception as e:
        raise Exception(f"Erro inesperado ao buscar dados: {str(e)}")


def calcular_metricas_energeticas(df: pd.DataFrame, temp_max: float) -> Dict[str, Optional[float]]:
    """Calcula indicadores térmicos chave usando temperatura interna e externa."""
    resultado: Dict[str, Optional[float]] = {
        "delta_t_medio": None,
        "delta_t_p95": None,
        "graus_hora_acima_limite": None,
        "slope_temp_ext_int": None,
        "corr_pearson": None,
        "lag_horas": None
    }

    if "temp_interna_media" not in df.columns or "temp_externa" not in df.columns:
        return resultado

    dados = df[["temp_interna_media", "temp_externa"]].dropna()
    if dados.empty:
        return resultado

    dados = dados.sort_index()
    delta_t = dados["temp_interna_media"] - dados["temp_externa"]
    resultado["delta_t_medio"] = float(delta_t.mean())
    resultado["delta_t_p95"] = float(delta_t.quantile(0.95))

    if isinstance(dados.index, pd.DatetimeIndex):
        dt_horas = dados.index.to_series().diff().dt.total_seconds().fillna(0) / 3600
        excedente = (dados["temp_interna_media"] - temp_max).clip(lower=0)
        resultado["graus_hora_acima_limite"] = float((excedente * dt_horas).sum())

        passo_medio_horas = dt_horas[dt_horas > 0].median()
        if pd.notna(passo_medio_horas) and passo_medio_horas > 0:
            int_vals = dados["temp_interna_media"].to_numpy()
            ext_vals = dados["temp_externa"].to_numpy()
            if len(int_vals) >= 3 and np.nanstd(int_vals) > 0 and np.nanstd(ext_vals) > 0:
                int_norm = int_vals - np.nanmean(int_vals)
                ext_norm = ext_vals - np.nanmean(ext_vals)
                corr = np.correlate(int_norm, ext_norm, mode="full")
                lags = np.arange(-len(int_norm) + 1, len(int_norm))
                melhor_idx = int(np.argmax(corr))
                lag_passos = lags[melhor_idx]
                resultado["lag_horas"] = float(lag_passos * passo_medio_horas)

    if dados["temp_externa"].nunique() > 1:
        slope, _, r_value, _, _ = stats.linregress(
            dados["temp_externa"],
            dados["temp_interna_media"]
        )
        resultado["slope_temp_ext_int"] = float(slope)
        resultado["corr_pearson"] = float(r_value)
    elif dados["temp_externa"].nunique() == 1 and dados["temp_interna_media"].nunique() == 1:
        resultado["slope_temp_ext_int"] = 0.0
        resultado["corr_pearson"] = 0.0

    return resultado


def render_metric_with_help(label: str, valor: Optional[str], ajuda: str, delta: Optional[str] = None):
    """Exibe métrica acompanhada de um popover com explicação."""
    col_metric, col_help = st.columns([4, 1])
    with col_metric:
        st.metric(label, valor if valor is not None else "—", delta=delta)
    with col_help:
        with st.popover("?", use_container_width=True):
            st.write(ajuda)

def extract_metadata_from_csv(csv_file: Path) -> Dict:
    """Extrai metadados e informações do CSV"""
    metadata = {
        "sensor_id": None,
        "modelo": None,
        "firmware": None,
        "tipo_sensor": None,
        "numero_viagem": None,
        "qualificacao": None,
        "fuso_horario": None,
        "intervalo_registro": None,
        "alarmes": {},
        "resumo": {},
        "arquivo_criado": None
    }
    
    try:
        sep, enc = sniff_delim_and_encoding(csv_file)
        sensor_id = parse_sensor_id(csv_file.name)
        metadata["sensor_id"] = sensor_id
        
        with open(csv_file, 'r', encoding=enc, errors='ignore') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Arquivo criado em
            if "arquivo criado em" in line_lower:
                date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
                if date_match:
                    metadata["arquivo_criado"] = date_match.group(1)
            
            # Modelo
            if "modelo do dispositivo" in line_lower:
                parts = line.split(sep)
                if len(parts) >= 2:
                    metadata["modelo"] = parts[1].strip()
            
            # Número de série (já temos do parse_sensor_id)
            
            # Tipo de Sensor
            if "tipo de sensor" in line_lower:
                parts = line.split(sep)
                for p in parts:
                    if "temperatura" in p.lower() and "umidade" in p.lower():
                        metadata["tipo_sensor"] = p.strip()
                        break
            
            # Firmware
            if "versão do firmware" in line_lower or "versao do firmware" in line_lower:
                parts = line.split(sep)
                for p in parts:
                    if "v" in p.lower() and re.search(r"v\d+", p.lower()):
                        metadata["firmware"] = p.strip()
                        break
            
            # Número da Viagem
            if "número da viagem" in line_lower and "000000" in line:
                parts = line.split(sep)
                for p in parts:
                    if re.match(r"\d{7}", p.strip()):
                        metadata["numero_viagem"] = p.strip()
            
            # Qualificação
            if "qualificacao" in line_lower or "qualifica" in line_lower:
                parts = line.split(sep)
                for p in parts:
                    if "qualificacao" in p.lower() or "qualifica" in p.lower():
                        metadata["qualificacao"] = p.strip()[:50]  # Limita tamanho
                        break
            
            # Fuso horário
            if "fuso horário" in line_lower or "fuso hor" in line_lower:
                parts = line.split(sep)
                for p in parts:
                    if "utc" in p.lower():
                        metadata["fuso_horario"] = p.strip()
                        break
            
            # Intervalo de registro
            if "intervalo de registro" in line_lower:
                parts = line.split(sep)
                for i, p in enumerate(parts):
                    if "intervalo" in p.lower():
                        if i + 1 < len(parts):
                            metadata["intervalo_registro"] = parts[i + 1].strip()
                            break
            
            # Alarmes H1, L1
            if "H1:" in line or "L1:" in line:
                parts = [p.strip() for p in line.split(sep) if p.strip()]
                if len(parts) >= 2:
                    tipo = parts[0].replace(":", "")
                    temp_match = re.search(r"(\d+[.,]\d+)", parts[1])
                    if temp_match:
                        temp_str = temp_match.group(1).replace(",", ".")
                        metadata["alarmes"][tipo] = {
                            "limite": float(temp_str),
                            "status": parts[-1] if len(parts) > 4 else None
                        }
            
            # Resumo - Máximo, Mínimo, Média
            if "máximo" in line_lower or "mximo" in line_lower:
                temp_match = re.search(r"(\d+[.,]\d+)°c", line, re.IGNORECASE)
                if temp_match:
                    metadata["resumo"]["max_temp"] = float(temp_match.group(1).replace(",", "."))
            
            if "mínimo" in line_lower or "mnimo" in line_lower:
                temp_match = re.search(r"(\d+[.,]\d+)°c", line, re.IGNORECASE)
                if temp_match:
                    metadata["resumo"]["min_temp"] = float(temp_match.group(1).replace(",", "."))
            
            if "média" in line_lower or "mdia" in line_lower:
                temp_match = re.search(r"(\d+[.,]\d+)°c", line, re.IGNORECASE)
                if temp_match:
                    metadata["resumo"]["media_temp"] = float(temp_match.group(1).replace(",", "."))
            
            # MKT
            if "mkt" in line_lower:
                temp_match = re.search(r"(\d+[.,]\d+)", line)
                if temp_match:
                    metadata["resumo"]["mkt"] = float(temp_match.group(1).replace(",", "."))
            
            # Leituras Atuais
            if "leituras atuais" in line_lower or "leituras a" in line_lower:
                num_match = re.search(r"(\d+)", line)
                if num_match:
                    metadata["resumo"]["leituras_atuais"] = int(num_match.group(1))
            
            # Primeira/Ãšltima leitura
            if "primeira leitura" in line_lower:
                date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", line)
                if date_match:
                    metadata["resumo"]["primeira_leitura"] = date_match.group(1)
            
            if "ultima leitura" in line_lower or "última leitura" in line_lower:
                date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
                if date_match:
                    metadata["resumo"]["ultima_leitura"] = date_match.group(1)
    
    except Exception as e:
        pass  # Ignora erros de extração de metadados
    
    return metadata

def load_internal_data(csv_dir: Path):
    """Carrega e processa dados dos sensores internos"""
    frames = []
    metadados_sensores = {}
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        return None, [], {}
    
    for csv_file in csv_files:
        try:
            sensor_id = parse_sensor_id(csv_file.name)
            
            # Extrai metadados do CSV
            metadata = extract_metadata_from_csv(csv_file)
            metadados_sensores[sensor_id] = metadata
            
            sep, enc = sniff_delim_and_encoding(csv_file)
            
            # Tenta ler com encoding detectado, se falhar tenta outros
            # Primeiro detecta onde começam os dados (linha com cabeçalho "Não.", "Tempo", etc.)
            skip_rows = None
            try:
                with open(csv_file, 'r', encoding=enc, errors='ignore') as f:
                    for i, raw_line in enumerate(f):
                        normalized = raw_line.replace('\ufeff', '')
                        cols = [
                            col.strip().lower()
                            for col in normalized.split(sep)
                            if col.strip()
                        ]
                        if not cols:
                            continue

                        cols_normalized = [
                            col.replace('°', '').replace('Âº', '').replace('ï¿½', '')
                            for col in cols
                        ]

                        has_tempo = any(col == 'tempo' for col in cols_normalized)
                        has_temp = any(
                            col.startswith('temperatura')
                            or col.startswith('temperaturac')
                            or (
                                col.startswith('temp')
                                and 'umidade' not in col
                                and 'umid' not in col
                                and 'rh' not in col
                            )
                            for col in cols_normalized
                        )

                        if has_tempo and has_temp:
                            skip_rows = i
                            break
            except Exception:
                pass
            
            df = None
            encodings_to_try = [enc, "latin-1", "cp1252", "iso-8859-1", "utf-8"]
            for encoding in encodings_to_try:
                try:
                    # Tenta com diferentes parâmetros dependendo da versão do pandas
                    read_params = {
                        'sep': sep,
                        'encoding': encoding,
                        'engine': 'python'
                    }
                    if skip_rows is not None:
                        read_params['skiprows'] = skip_rows
                    
                    try:
                        read_params['on_bad_lines'] = 'skip'
                        df = pd.read_csv(csv_file, **read_params)
                    except TypeError:
                        # Versão antiga do pandas não tem on_bad_lines
                        read_params.pop('on_bad_lines', None)
                        read_params['error_bad_lines'] = False
                        df = pd.read_csv(csv_file, **read_params)
                    break
                except (UnicodeDecodeError, Exception) as e:
                    if encoding == encodings_to_try[-1]:  # Último encoding
                        raise e
                    continue
            
            if df is None or df.empty:
                continue
                
            df.columns = [str(c).strip() for c in df.columns]
            
            # Detecta timestamp (prioriza "Tempo" que é o nome da coluna nos CSVs)
            ts_col = None
            for c in df.columns:
                cl = c.lower().strip()
                # Prioriza "tempo" que é o nome exato da coluna nos CSVs
                if cl == "tempo":
                    ts_col = c
                    break
                elif any(k in cl for k in ["timestamp", "data e hora", "data/hora", "datahora", "data", "date", "hora", "time"]):
                    ts_col = c
                    break
            if ts_col is None:
                ts_col = df.columns[0]
            
            ts_raw = df[ts_col].astype(str)
            ts = parse_timestamp_series(ts_raw)
            
            # Detecta temperatura - PRIORIZA explicitamente "Temperatura°C"
            # EVITA colunas de umidade (Umidade%RH)
            t_col = None
            
            # ESTRATÉGIA: Primeiro identifica TODAS as colunas candidatas
            # Depois valida pelos VALORES (não só pelo nome)
            candidatas_temp = []
            candidatas_umid = []
            
            for c in df.columns:
                if c == ts_col:
                    continue
                cl = c.lower().strip()
                c_orig = str(c).strip()
                
                # Identifica colunas de temperatura - mesmo com encoding corrompido
                # "Temperatura°C" pode aparecer como "TemperaturaC", "TemperaturaC", etc.
                # Remove caracteres não-ASCII para comparação mais robusta
                c_normalized = c_orig.replace('°', '').replace('', '').replace('\ufeff', '').strip()
                cl_normalized = c_normalized.lower()
                
                # Identifica colunas de temperatura (mesmo com ° corrompido)
                # Procura por "temperatura" seguido opcionalmente por "c" ou "°c"
                has_temp = (
                    "temperatura" in cl or 
                    "temperatura" in cl_normalized or 
                    "temperaturac" in cl_normalized or
                    ("temp" in cl and "umidade" not in cl and "umid" not in cl and "rh" not in cl)
                )
                no_umidade = "umidade" not in cl and "umid" not in cl and "rh" not in cl
                
                if has_temp and no_umidade:
                    candidatas_temp.append(c)
                
                # Identifica colunas de umidade explicitamente
                if "umidade" in cl or "umid" in cl or "rh" in cl:
                    candidatas_umid.append(c)
            
            # VALIDAÇÃO POR VALORES: Testa cada candidata e escolhe a melhor
            melhor_col = None
            melhor_score = -1
            
            # Se não encontrou candidatas por nome, adiciona TODAS as colunas (exceto umidade e timestamp)
            if not candidatas_temp:
                for c in df.columns:
                    if c != ts_col:
                        col_cl = c.lower().strip()
                        if "umidade" not in col_cl and "umid" not in col_cl and "rh" not in col_cl:
                            candidatas_temp.append(c)
            
            for col_candidata in candidatas_temp:
                try:
                    valores = pd.to_numeric(
                        df[col_candidata].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                        errors="coerce"
                    ).dropna()
                    
                    if len(valores) > 0:
                        media = valores.mean()
                        min_val = valores.min()
                        max_val = valores.max()
                        
                        # Score: melhor se média está entre 15-40°C (temperatura ambiente)
                        # Pior se média está entre 50-100 (umidade)
                        if 15 <= media <= 40 and min_val >= 10 and max_val <= 50:
                            score = 100 - abs(media - 30)  # Melhor quanto mais próximo de 30°C
                            if score > melhor_score:
                                melhor_score = score
                                melhor_col = col_candidata
                except:
                    continue
            
            # Se não encontrou candidata boa, tenta TODAS as colunas (exceto umidade e timestamp)
            # Isso garante que mesmo se o nome estiver corrompido, encontra pelos valores
            if melhor_col is None:
                for col_candidata in df.columns:
                    if col_candidata == ts_col:
                        continue
                    
                    # Pula colunas conhecidamente de umidade
                    col_cl = col_candidata.lower().strip()
                    if "umidade" in col_cl or "umid" in col_cl or "rh" in col_cl:
                        continue
                    
                    try:
                        valores = pd.to_numeric(
                            df[col_candidata].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                            errors="coerce"
                        ).dropna()
                        
                        if len(valores) > 0:
                            media = valores.mean()
                            min_val = valores.min()
                            max_val = valores.max()
                            
                            # Aceita se está na faixa de temperatura (15-40°C)
                            # E rejeita se está na faixa de umidade (50-100)
                            if 15 <= media <= 40 and min_val >= 10 and max_val <= 50 and media < 50:
                                melhor_col = col_candidata
                                break
                    except:
                        continue
            
            t_col = melhor_col
            
            # Validação crítica: verifica se a coluna detectada realmente é temperatura
            if t_col is None:
                # ÚLTIMO RECURSO: Tenta todas as colunas, mesmo sem validação de nome
                # Isso resolve problemas de encoding corrompido
                for col_candidata in df.columns:
                    if col_candidata == ts_col:
                        continue
                    
                    col_cl = col_candidata.lower().strip()
                    # Pula apenas se for OBVIAMENTE umidade
                    if "umidade" in col_cl or ("umid" in col_cl and "rh" in col_cl):
                        continue
                    
                    try:
                        valores = pd.to_numeric(
                            df[col_candidata].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                            errors="coerce"
                        ).dropna()
                        
                        if len(valores) > 10:  # Precisa ter pelo menos alguns valores
                            media = valores.mean()
                            min_val = valores.min()
                            max_val = valores.max()
                            
                            # Se os valores estão na faixa de temperatura (10-50°C) E não são umidade (>50)
                            if 10 <= media <= 50 and min_val >= 5 and max_val <= 60 and media < 50:
                                t_col = col_candidata
                                st.info(f"â„¹ï¸ Detectada coluna '{col_candidata}' como temperatura em {csv_file.name} (média: {media:.1f}°C)")
                                break
                    except:
                        continue
            
            # Se AINDA não encontrou, tenta pela POSIÇÃO (geralmente temperatura é a 3ª coluna: Não., Tempo, Temperatura)
            if t_col is None and len(df.columns) >= 3:
                # Tenta a terceira coluna (depois de "Não." e "Tempo", geralmente é "Temperatura°C")
                colunas_ordenadas = [c for c in df.columns if c != ts_col]
                # Pula a primeira coluna (geralmente "Não.") e pega a segunda (geralmente temperatura)
                if len(colunas_ordenadas) >= 2:
                    col_teste = colunas_ordenadas[1]  # Segunda coluna após timestamp (índice 1)
                elif len(colunas_ordenadas) >= 1:
                    col_teste = colunas_ordenadas[0]  # Primeira coluna disponível
                else:
                    col_teste = None
                
                if col_teste:
                    try:
                        valores = pd.to_numeric(
                            df[col_teste].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                            errors="coerce"
                        ).dropna()
                        if len(valores) > 10:
                            media = valores.mean()
                            min_val = valores.min()
                            max_val = valores.max()
                            # Valida que não é umidade (umidade geralmente tem média > 50)
                            if 10 <= media <= 50 and min_val >= 5 and max_val <= 60:
                                t_col = col_teste
                                st.info(f"â„¹ï¸ Usando coluna '{col_teste}' como temperatura (fallback por posição) em {csv_file.name} (média: {media:.1f}°C)")
                    except:
                        pass
            
            if t_col is None:
                st.warning(f"âš ï¸ Não foi possível detectar coluna de temperatura em {csv_file.name}. Colunas disponíveis: {df.columns.tolist()}")
                continue
            
            # Verifica se a coluna detectada não é umidade
            t_col_lower = t_col.lower().strip()
            if "umidade" in t_col_lower or "umid" in t_col_lower or "rh" in t_col_lower:
                # Se detectou umidade por engano, procura novamente excluindo esta
                t_col = None
                for c in df.columns:
                    cl = c.lower().strip()
                    if c == ts_col:
                        continue
                    if "temperatura" in cl or ("temp" in cl and "umidade" not in cl and "umid" not in cl and "rh" not in cl):
                        t_col = c
                        break
                if t_col is None:
                    st.warning(f"âš ï¸ Apenas coluna de umidade encontrada em {csv_file.name}, pulando arquivo")
                    continue
            
            # Extrai valores da coluna detectada
            temp_raw = pd.to_numeric(
                df[t_col].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                errors="coerce"
            )
            
            # VALIDAÇÃO CRÍTICA ANTES DE USAR: Verifica se os valores fazem sentido para temperatura
            temp_validos = temp_raw.dropna()
            
            if len(temp_validos) == 0:
                st.warning(f"âš ï¸ Nenhum valor válido na coluna '{t_col}' em {csv_file.name}")
                continue
            
            temp_media = temp_validos.mean()
            temp_min = temp_validos.min()
            temp_max = temp_validos.max()
            
            # VALIDAÇÃO: Se os valores estão claramente na faixa de umidade (40-100), TROCA de coluna
            if temp_min >= 40 and temp_max <= 100 and temp_media > 50:
                # Procura TODAS as outras colunas que possam ser temperatura
                coluna_corrigida = False
                for alt_col in df.columns:
                    if alt_col == t_col or alt_col == ts_col:
                        continue
                    
                    alt_cl = alt_col.lower().strip()
                    # REJEITA explicitamente colunas de umidade
                    if "umidade" in alt_cl or "umid" in alt_cl or "rh" in alt_cl:
                        continue
                    
                    # Tenta esta coluna alternativa
                    alt_temp = pd.to_numeric(
                        df[alt_col].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                        errors="coerce"
                    ).dropna()
                    
                    if len(alt_temp) > 0:
                        alt_media = alt_temp.mean()
                        alt_min = alt_temp.min()
                        alt_max = alt_temp.max()
                        
                        # Se a alternativa tem valores razoáveis para temperatura (10-50°C)
                        # e a média está abaixo de 50 (não é umidade)
                        if 10 <= alt_min and alt_max <= 50 and alt_media < 50:
                            # Esta é provavelmente a coluna correta!
                            t_col = alt_col
                            temp_raw = alt_temp
                            coluna_corrigida = True
                            st.success(f"✅ Corrigido automaticamente: '{csv_file.name}' - usando coluna '{alt_col}' (temperatura: {alt_media:.1f}°C) ao invés de '{t_col}' (umidade: {temp_media:.1f}%)")
                            break
                
                # Se não conseguiu corrigir e ainda parece umidade, rejeita
                if not coluna_corrigida:
                    st.error(f"âŒ ERRO: Coluna '{t_col}' contém valores de umidade (média: {temp_media:.1f}%) em {csv_file.name}. Pulando arquivo. Verifique se a coluna 'Temperatura°C' existe no CSV.")
                    continue
            
            # Usa a temperatura validada
            temp = temp_raw
            # Detecta umidade relativa associada ao sensor (se disponível)
            u_col = None
            melhor_qtd = 0
            if candidatas_umid:
                for col_umid in candidatas_umid:
                    if col_umid == t_col or col_umid == ts_col:
                        continue
                    try:
                        valores_umid = pd.to_numeric(
                            df[col_umid].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                            errors="coerce"
                        ).dropna()
                        if len(valores_umid) == 0:
                            continue
                        umid_min = valores_umid.min()
                        umid_max = valores_umid.max()
                        umid_media = valores_umid.mean()
                        if 0 <= umid_min <= 100 and 0 < umid_max <= 100 and 0 < umid_media <= 100:
                            if len(valores_umid) > melhor_qtd:
                                melhor_qtd = len(valores_umid)
                                u_col = col_umid
                    except Exception:
                        continue

            umid_series = None
            if u_col:
                umid_series = pd.to_numeric(
                    df[u_col].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                    errors="coerce"
                )
                if umid_series.notna().sum() == 0:
                    umid_series = None
                else:
                    metadados_sensores[sensor_id]["tem_umidade"] = True
                    metadados_sensores[sensor_id]["coluna_umidade_original"] = u_col

            cur = pd.DataFrame({
                "timestamp": ts,
                sensor_id: temp
            })
            if umid_series is not None:
                cur[f"{sensor_id}_umidade"] = umid_series
            cur = cur[~cur["timestamp"].isna()].sort_values("timestamp")
            
            # Garante que timestamp é datetime64
            cur["timestamp"] = pd.to_datetime(cur["timestamp"], errors="coerce")
            cur = cur[~cur["timestamp"].isna()]
            
            # Remove valores inválidos
            cur = cur[cur[sensor_id].notna()]
            
            if len(cur) == 0:
                continue
            
            frames.append(cur[["timestamp", sensor_id]])
            
        except Exception as e:
            st.warning(f"Erro ao processar {csv_file.name}: {e}")
            continue
    
    if not frames:
        return None, [], {}
    
    from functools import reduce
    wide = reduce(lambda L, R: pd.merge(L, R, on="timestamp", how="outer"), frames)
    
    # Garante que timestamp é datetime antes de definir como índice
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], errors="coerce")
    wide = wide[~wide["timestamp"].isna()]
    wide = wide.set_index("timestamp").sort_index()
    
    # Garante que o índice é DatetimeIndex
    if not isinstance(wide.index, pd.DatetimeIndex):
        wide.index = pd.to_datetime(wide.index, errors="coerce")
        wide = wide[~wide.index.isna()]
    
    sensor_cols = [
        c for c in wide.columns
        if c.startswith("EL") and not c.lower().endswith("_umidade")
    ]
    wide["temp_interna_media"] = wide[sensor_cols].mean(axis=1)
    wide["temp_interna_min"] = wide[sensor_cols].min(axis=1)
    wide["temp_interna_max"] = wide[sensor_cols].max(axis=1)
    wide["temp_interna_std"] = wide[sensor_cols].std(axis=1)
    
    return wide, sensor_cols, metadados_sensores

def load_external_data(filepath: Path):
    """Carrega dados de temperatura externa"""
    if not filepath.exists():
        return None
    
    sep, enc = sniff_delim_and_encoding(filepath)
    df = pd.read_csv(filepath, sep=sep, encoding=enc)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    ts_col = None
    for c in df.columns:
        if any(k in c for k in ["timestamp", "data", "date", "hora", "time"]):
            ts_col = c
            break
    
    temp_col = None
    for c in df.columns:
        if any(k in c for k in ["temp", "temperatura", "celsius", "°c"]):
            if "extern" in c or ts_col is None or c != ts_col:
                temp_col = c
                break
    
    if ts_col is None or temp_col is None:
        return None
    
    ts = parse_timestamp_series(df[ts_col])
    temp = pd.to_numeric(
        df[temp_col].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
        errors="coerce"
    )
    
    result = pd.DataFrame({
        "timestamp": ts,
        "temp_externa": temp
    }).dropna().set_index("timestamp").sort_index()
    
    return result


def load_scenario_dataset(label: str, csv_dir: Path, location: Dict) -> Optional[Dict]:
    """Carrega dados internos e busca automaticamente temperatura externa para um cenário."""
    df, sensor_cols, metadata = load_internal_data(csv_dir)
    if df is None or len(sensor_cols) == 0:
        st.warning(f"⚠️ Nenhum dado encontrado para {label} em {csv_dir}")
        return None

    df = df.sort_index()
    start_ts, end_ts = df.index.min(), df.index.max()

    external_df = None
    try:
        external_df = fetch_external_temperature_from_api(
            latitude=location["latitude"],
            longitude=location["longitude"],
            start_ts=start_ts,
            end_ts=end_ts,
            tz=location.get("timezone")
        )
    except Exception as exc:
        st.warning(f"⚠️ Falha ao obter temperatura externa para {label}: {exc}")

    if external_df is not None and not external_df.empty:
        df = df.join(external_df, how="outer").sort_index()
        if "temp_externa" in df.columns:
            df["temp_externa"] = (
                df["temp_externa"]
                .interpolate(method="time")
                .ffill()
                .bfill()
            )

    return {
        "label": label,
        "df": df,
        "sensor_cols": sensor_cols,
        "metadata": metadata,
        "start": start_ts,
        "end": end_ts,
    }


def filter_scenario_dataframe(
    df: pd.DataFrame,
    sensor_cols: List[str],
    selected_sensors: List[str],
    date_range: Optional[Tuple[datetime, datetime]]
) -> Tuple[pd.DataFrame, List[str]]:
    """Aplica filtros de data e sensores ao dataframe do cenário."""
    filtered = df.copy()
    if isinstance(filtered.index, pd.DatetimeIndex) and date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered.index.date >= start_date) &
            (filtered.index.date <= end_date)
        ]

    scenario_sensors = [s for s in selected_sensors if s in sensor_cols]
    if scenario_sensors:
        filtered["temp_interna_media"] = filtered[scenario_sensors].mean(axis=1)
        filtered["temp_interna_min"] = filtered[scenario_sensors].min(axis=1)
        filtered["temp_interna_max"] = filtered[scenario_sensors].max(axis=1)
        filtered["temp_interna_std"] = filtered[scenario_sensors].std(axis=1)
    else:
        filtered["temp_interna_media"] = np.nan
        filtered["temp_interna_min"] = np.nan
        filtered["temp_interna_max"] = np.nan
        filtered["temp_interna_std"] = np.nan

    return filtered, scenario_sensors


def compute_degree_hours_from_df(df: pd.DataFrame, column: str, limit: float) -> Optional[float]:
    """Calcula graus-hora acima do limite para uma coluna específica do dataframe."""
    if column not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
        return None
    serie = pd.to_numeric(df[column], errors="coerce")
    if serie.dropna().empty:
        return None
    dt_hours = df.index.to_series().diff().dt.total_seconds().fillna(0) / 3600.0
    excedente = (serie - limit).clip(lower=0).fillna(0)
    return float((excedente * dt_hours).sum())


def summarize_scenario(df: pd.DataFrame, temp_max: float) -> Dict[str, Optional[float]]:
    """Resumo rápido para comparação entre cenários com métricas normalizadas."""
    summary: Dict[str, Optional[float]] = {
        "media_interna": None,
        "min_interna": None,
        "max_interna": None,
        "pct_tempo_acima": None,
        "corr_pearson": None,
        "gradiente_medio": None,
        "graus_hora": None,
        "media_delta": None,
        "pct_ext_acima": None,
        "graus_hora_ext": None,
        "ratio_excursoes": None,
        "graus_hora_norm": None,
        "temp_externa_media": None,
        "amplitude_interna": None,
        "amplitude_externa": None,
        "attenuation_factor": None,
        "std_interna": None,
        "std_externa": None,
        "stability_index": None,
    }
    if df is None or df.empty or "temp_interna_media" not in df.columns:
        return summary

    valores = df["temp_interna_media"].dropna()
    if valores.empty:
        return summary

    summary["media_interna"] = float(valores.mean())
    summary["min_interna"] = float(valores.min())
    summary["max_interna"] = float(valores.max())
    summary["amplitude_interna"] = float(valores.max() - valores.min())
    summary["std_interna"] = float(valores.std(ddof=0))
    acima = (valores > temp_max).sum()
    summary["pct_tempo_acima"] = float(100 * acima / len(valores))

    if "temp_externa" in df.columns:
        pares = df[["temp_interna_media", "temp_externa"]].dropna()
        if len(pares) > 10:
            gradiente = pares["temp_externa"] - pares["temp_interna_media"]
            summary["gradiente_medio"] = float(gradiente.mean())
            summary["corr_pearson"] = float(pares["temp_interna_media"].corr(pares["temp_externa"]))
            summary["media_delta"] = float((-gradiente).mean())  # interna - externa
        ext = df["temp_externa"].dropna()
        if not ext.empty:
            summary["temp_externa_media"] = float(ext.mean())
            pct_ext = float(100 * (ext > temp_max).sum() / len(ext))
            summary["pct_ext_acima"] = pct_ext
            summary["amplitude_externa"] = float(ext.max() - ext.min())
            summary["std_externa"] = float(ext.std(ddof=0))
            if summary["amplitude_externa"] and summary["amplitude_externa"] > 0:
                summary["attenuation_factor"] = summary["amplitude_interna"] / summary["amplitude_externa"]
            if summary["std_externa"] and summary["std_externa"] > 0:
                summary["stability_index"] = summary["std_interna"] / summary["std_externa"]

    metricas = calcular_metricas_energeticas(df, temp_max)
    summary["graus_hora"] = metricas.get("graus_hora_acima_limite")

    if "temp_externa" in df.columns:
        graus_hora_ext = compute_degree_hours_from_df(df, "temp_externa", temp_max)
        summary["graus_hora_ext"] = graus_hora_ext
        if summary["pct_tempo_acima"] is not None and summary["pct_ext_acima"] and summary["pct_ext_acima"] > 0:
            summary["ratio_excursoes"] = summary["pct_tempo_acima"] / summary["pct_ext_acima"]
        if summary["graus_hora"] is not None and graus_hora_ext and graus_hora_ext > 0:
            summary["graus_hora_norm"] = summary["graus_hora"] / graus_hora_ext

    return summary


def render_scenario_section(
    labels: List[str],
    renderer,
    scenario_map: Dict[str, Dict],
    share_limits: bool = False
):
    """Renderiza conteúdo para um ou dois cenários lado a lado."""
    if len(labels) == 1:
        renderer(labels[0], st, scenario_map, limits=None, gather_only=False)
        return

    if not share_limits:
        columns = st.columns(len(labels))
        for column, label in zip(columns, labels):
            with column:
                st.caption(f"Cenário: **{label}**")
                renderer(label, column, scenario_map, limits=None, gather_only=False)
        return

    # Calcula limites comuns APENAS para eixo Y
    y_mins, y_maxs = [], []
    for label in labels:
        scenario_limits = renderer(label, None, scenario_map, limits=None, gather_only=True)
        if not scenario_limits:
            continue
        y_range = scenario_limits.get("y")
        if y_range:
            y_mins.append(y_range[0])
            y_maxs.append(y_range[1])
    common_limits = None
    if y_mins and y_maxs:
        common_limits = {"y": (min(y_mins), max(y_maxs))}

    columns = st.columns(len(labels))
    for column, label in zip(columns, labels):
        with column:
            st.caption(f"Cenário: **{label}**")
            renderer(label, column, scenario_map, limits=common_limits, gather_only=False)

# ============================================================================
# FUNÃ‡Ã•ES DE VISUALIZAÃ‡ÃƒO
# ============================================================================

def plot_temperature_over_time(df, sensor_cols, show_external=True):
    """Gráfico de temperatura ao longo do tempo"""
    # Garante que o índice é DatetimeIndex (cria cópia para não modificar original)
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Temperatura Interna - Todos os Sensores", "Temperatura Interna - Média e Faixa"),
        row_heights=[0.6, 0.4]
    )
    
    # Gráfico 1: Todos os sensores
    colors = px.colors.qualitative.Set3
    for i, sensor in enumerate(sensor_cols):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[sensor],
                name=sensor,
                mode='lines',
                line=dict(width=1, color=colors[i % len(colors)]),
                opacity=0.6,
                hovertemplate=f'{sensor}<br>%{{x}}<br>%{{y:.2f}}°C<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Linha de referência 30°C
    fig.add_hline(
        y=30, line_dash="dash", line_color="red",
        annotation_text="Limite 30°C", annotation_position="right",
        row=1, col=1
    )
    
    # Gráfico 2: Média e faixa
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["temp_interna_media"],
            name="Média",
            mode='lines',
            line=dict(width=2, color='blue'),
            hovertemplate='Média<br>%{x}<br>%{y:.2f}°C<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["temp_interna_min"],
            name="Mínima",
            mode='lines',
            line=dict(width=1, color='lightblue', dash='dash'),
            hovertemplate='Mínima<br>%{x}<br>%{y:.2f}°C<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["temp_interna_max"],
            name="Máxima",
            mode='lines',
            line=dict(width=1, color='lightblue', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)',
            hovertemplate='Máxima<br>%{x}<br>%{y:.2f}°C<extra></extra>'
        ),
        row=2, col=1
    )
    
    if show_external and "temp_externa" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["temp_externa"],
                name="Temperatura Externa",
                mode='lines',
                line=dict(width=2, color='orange', dash='dot'),
                hovertemplate='Externa<br>%{x}<br>%{y:.2f}°C<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.add_hline(
        y=30, line_dash="dash", line_color="red",
        annotation_text="Limite 30°C", annotation_position="right",
        row=2, col=1
    )
    
    fig.update_xaxes(
        title_text="Data/Hora", 
        row=2, col=1,
        type="date",  # Força tipo de data no eixo X
        tickformat="%d/%m/%Y %H:%M"  # Formato brasileiro
    )
    fig.update_xaxes(
        type="date",  # Força tipo de data no eixo X superior também
        tickformat="%d/%m/%Y %H:%M",
        row=1, col=1
    )
    fig.update_yaxes(title_text="Temperatura (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Temperatura (°C)", row=2, col=1)
    fig.update_layout(
        height=700, 
        showlegend=True, 
        hovermode='x unified',
        xaxis=dict(type='date'),  # Garante tipo date no layout principal
        xaxis2=dict(type='date')   # Garante tipo date no segundo subplot
    )
    
    return fig

def plot_correlation_scatter(df, temp_int_col, temp_ext_col):
    """Gráfico de dispersão de correlação"""
    data = df[[temp_int_col, temp_ext_col]].dropna()
    
    if len(data) < 10:
        return None
    
    # Calcula correlação
    corr, p_value = stats.pearsonr(data[temp_ext_col], data[temp_int_col])
    
    # Regressão linear
    slope, intercept, r_value, _, _ = stats.linregress(data[temp_ext_col], data[temp_int_col])
    x_line = np.linspace(data[temp_ext_col].min(), data[temp_ext_col].max(), 100)
    y_line = slope * x_line + intercept
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=data[temp_ext_col],
            y=data[temp_int_col],
            mode='markers',
            marker=dict(
                size=4,
                color=data.index.astype(int),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Timestamp")
            ),
            hovertemplate='Externa: %{x:.2f}°C<br>Interna: %{y:.2f}°C<extra></extra>',
            name="Medições"
        )
    )
    
    # Linha de regressão
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Regressão (R²={r_value**2:.3f})'
        )
    )
    
    # Linha diagonal (y=x)
    min_temp = min(data[temp_ext_col].min(), data[temp_int_col].min())
    max_temp = max(data[temp_ext_col].max(), data[temp_int_col].max())
    fig.add_trace(
        go.Scatter(
            x=[min_temp, max_temp],
            y=[min_temp, max_temp],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name='y=x'
        )
    )
    
    fig.update_layout(
        title=f'Correlação: Externa vs Interna<br>Pearson r = {corr:.3f} (p = {p_value:.2e})',
        xaxis_title='Temperatura Externa (°C)',
        yaxis_title='Temperatura Interna Média (°C)',
        height=500,
        hovermode='closest'
    )
    
    return fig

def plot_thermal_gradient(df, temp_int_col, temp_ext_col):
    """Gráfico de gradiente térmico ao longo do tempo"""
    # Garante que o índice é DatetimeIndex (cria cópia para não modificar original)
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
    
    data = df[[temp_int_col, temp_ext_col]].dropna()
    data["gradiente"] = data[temp_ext_col] - data[temp_int_col]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["gradiente"],
            mode='lines',
            name="Gradiente Térmico",
            line=dict(width=2, color='green'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)',
            hovertemplate='%{x}<br>Gradiente: %{y:.2f}°C<extra></extra>'
        )
    )
    
    # Linha de referência zero
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Linhas de referência para eficiência
    fig.add_hline(y=2, line_dash="dot", line_color="orange", annotation_text="Bom isolamento")
    fig.add_hline(y=5, line_dash="dot", line_color="green", annotation_text="Excelente isolamento")
    
    fig.update_layout(
        title="Gradiente Térmico (Externa - Interna)",
        xaxis_title="Data/Hora",
        yaxis_title="Gradiente (°C)",
        height=400,
        hovermode='x unified',
        xaxis=dict(
            type='date',
            tickformat="%d/%m/%Y %H:%M"
        )
    )
    
    return fig

def plot_heatmap_by_period(df, temp_int_col):
    """Mapa de calor por período do dia"""
    data = df[[temp_int_col]].copy().dropna()
    data["hora"] = data.index.hour
    data["dia_semana"] = data.index.strftime("%A")
    data["dia"] = data.index.date
    
    # Cria matriz para heatmap
    pivot_data = data.pivot_table(
        values=temp_int_col,
        index="hora",
        columns="dia",
        aggfunc="mean"
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[str(d) for d in pivot_data.columns],
        y=pivot_data.index,
        colorscale='RdYlBu_r',
        colorbar=dict(title="Temperatura (°C)"),
        hovertemplate='Hora: %{y}<br>Dia: %{x}<br>Temp: %{z:.2f}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title="Mapa de Calor - Temperatura por Hora e Dia",
        xaxis_title="Data",
        yaxis_title="Hora do Dia",
        height=400
    )
    
    return fig

def plot_sensor_comparison(df, sensor_cols):
    """Boxplot comparativo dos sensores"""
    data = []
    labels = []
    for sensor in sensor_cols:
        values = df[sensor].dropna()
        if len(values) > 0:
            data.append(values)
            labels.append(sensor)
    
    fig = go.Figure()
    
    for i, (values, label) in enumerate(zip(data, labels)):
        fig.add_trace(
            go.Box(
                y=values,
                name=label,
                boxpoints='outliers',
                hovertemplate=f'{label}<br>%{{y:.2f}}°C<extra></extra>'
            )
        )
    
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Limite 30°C")
    
    fig.update_layout(
        title="Distribuição de Temperatura por Sensor",
        xaxis_title="Sensor",
        yaxis_title="Temperatura (°C)",
        height=500
    )
    
    return fig

def plot_excursions_over_time(df, sensor_cols, threshold=30):
    """Gráfico de excursões acima do limite"""
    fig = go.Figure()
    
    for sensor in sensor_cols:
        data = df[[sensor]].dropna()
        excursions = (data[sensor] > threshold).astype(int)
        # Agrupa por dia mantendo o índice como datetime
        excursions_daily = excursions.groupby(excursions.index.date).sum()
        # Converte as datas de volta para datetime para plotagem
        dates = pd.to_datetime(excursions_daily.index)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=excursions_daily.values,
                mode='lines+markers',
                name=sensor,
                hovertemplate=f'{sensor}<br>%{{x}}<br>%{{y}} minutos acima de {threshold}°C<extra></extra>'
            )
        )
    
    fig.update_layout(
        title=f"Excursões Acima de {threshold}°C por Dia",
        xaxis_title="Data",
        yaxis_title="Minutos acima do limite",
        height=400,
        hovermode='x unified',
        xaxis=dict(
            type='date',
            tickformat="%d/%m/%Y"
        )
    )
    
    return fig

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    st.title("Dashboard de Análise Térmica - Tecnologia 3TC")
    st.markdown("---")
    
    st.sidebar.header("Configurações")
    temp_min = st.sidebar.number_input("Temperatura mínima ideal (°C)", value=15.0)
    temp_max = st.sidebar.number_input("Temperatura máxima ideal (°C)", value=30.0)

    scenario_configs = [
        ("Pré 3TC", PRE_INSTALL_DIR),
        ("Pós 3TC", POST_INSTALL_DIR),
    ]

    scenarios: Dict[str, Dict] = {}
    with st.spinner("Carregando cenários pré e pós..."):
        for label, directory in scenario_configs:
            scenario = load_scenario_dataset(label, directory, DEFAULT_LOCATION)
            if scenario:
                scenarios[label] = scenario

    if not scenarios:
        st.error("Não foi possível carregar nenhum cenário. Verifique as pastas fixas de dados.")
        return

    scenario_labels = list(scenarios.keys())
    overall_start = min(s["start"].date() for s in scenarios.values())
    overall_end = max(s["end"].date() for s in scenarios.values())

    st.sidebar.markdown("---")
    st.sidebar.header("Filtros Globais")
    date_range = st.sidebar.date_input(
        "Período de Análise",
        value=(overall_start, overall_end),
        min_value=overall_start,
        max_value=overall_end
    )

    all_sensors = sorted({sensor for s in scenarios.values() for sensor in s["sensor_cols"]})
    selected_sensors = st.sidebar.multiselect(
        "Sensores considerados",
        options=all_sensors,
        default=all_sensors
    )

    current_scenario_label = st.sidebar.selectbox(
        "Cenário exibido nas abas",
        options=scenario_labels
    )
    comparison_labels = st.sidebar.multiselect(
        "Cenários na comparação",
        options=scenario_labels,
        default=scenario_labels
    )

    filtered_scenarios: Dict[str, Dict] = {}
    for label, scenario in scenarios.items():
        filtered_df, active_sensors = filter_scenario_dataframe(
            scenario["df"],
            scenario["sensor_cols"],
            selected_sensors,
            date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None
        )
        filtered_scenarios[label] = {
            **scenario,
            "df": filtered_df,
            "active_sensors": active_sensors,
            "summary": summarize_scenario(filtered_df, temp_max)
        }

    current = filtered_scenarios[current_scenario_label]
    current_df = current["df"]
    current_sensors = current["active_sensors"]
    metadados_sensores = current.get("metadata", {})

    if current_df.empty or current_df["temp_interna_media"].dropna().empty:
        st.error(f"Não há dados válidos para o cenário {current_scenario_label} dentro dos filtros selecionados.")
        return

    dual_labels: List[str]
    if (
        "Pré 3TC" in filtered_scenarios
        and "Pós 3TC" in filtered_scenarios
        and not filtered_scenarios["Pré 3TC"]["df"].empty
        and not filtered_scenarios["Pós 3TC"]["df"].empty
    ):
        dual_labels = ["Pré 3TC", "Pós 3TC"]
    else:
        dual_labels = [current_scenario_label]
    
    st.caption(
        f"Cenário atual: **{current_scenario_label}** "
        f"({current['start'].strftime('%d/%m/%Y %H:%M')} → {current['end'].strftime('%d/%m/%Y %H:%M')})"
    )
    if len(dual_labels) == 2:
        st.info("Visualizações em duas colunas comparam diretamente o período **Pré 3TC** (esquerda) e **Pós 3TC** (direita).")
    
    st.header("Visão Geral")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sensores selecionados", len(current_sensors))
    with col2:
        st.metric("Total de medições", f"{len(current_df):,}")
    with col3:
        if "temp_interna_media" in current_df.columns:
            st.metric("Temperatura média interna", f"{current_df['temp_interna_media'].mean():.2f}°C")
    with col4:
        if "temp_interna_media" in current_df.columns:
            st.metric("Excursões > limite", int((current_df["temp_interna_media"] > temp_max).sum()))
    
    # Botão de exportar (último do menu)
    st.sidebar.markdown("---")
    exportar_excel = st.sidebar.button("📤 Exportar medições (Excel)", use_container_width=True)
    
    # Lógica de exportação (executada DEPOIS do botão ser criado)
    if exportar_excel:
        scenario_sensor_cols = [s for s in current["sensor_cols"] if s in current_df.columns]
        if current_df is not None and scenario_sensor_cols:
            hum_map = {
                sensor: f"{sensor}_umidade"
                for sensor in scenario_sensor_cols
                if f"{sensor}_umidade" in current_df.columns
            }
            temp_long = (
                current_df[scenario_sensor_cols]
                .reset_index()
                .rename(columns={"index": "timestamp"})
                .melt(id_vars="timestamp", var_name="sensor", value_name="temperatura")
                .dropna(subset=["temperatura"])
            )
            if temp_long.empty:
                st.sidebar.warning("Não há medições válidas para exportar.")
            else:
                # Adiciona temperatura externa se disponível
                if "temp_externa" in current_df.columns:
                    temp_externa_df = (
                        current_df[["temp_externa"]]
                        .reset_index()
                        .rename(columns={"index": "timestamp"})
                        .dropna(subset=["temp_externa"])
                    )
                    # Merge com temperatura externa por timestamp
                    export_df = temp_long.merge(
                        temp_externa_df,
                        on="timestamp",
                        how="left"
                    )
                else:
                    export_df = temp_long.copy()
                    export_df["temp_externa"] = pd.NA
                
                # Adiciona umidade se disponível
                if hum_map:
                    hum_cols = list(hum_map.values())
                    rename_map = {col: sensor for sensor, col in hum_map.items()}
                    hum_long = (
                        current_df[hum_cols]
                        .reset_index()
                        .rename(columns={"index": "timestamp"})
                        .rename(columns=rename_map)
                        .melt(id_vars="timestamp", var_name="sensor", value_name="umidade")
                    )
                    export_df = export_df.merge(hum_long, on=["timestamp", "sensor"], how="left")
                else:
                    export_df["umidade"] = pd.NA
                
                # Reordena colunas: timestamp, sensor, temperatura, temp_externa, umidade
                col_order = ["timestamp", "sensor", "temperatura"]
                if "temp_externa" in export_df.columns:
                    col_order.append("temp_externa")
                if "umidade" in export_df.columns:
                    col_order.append("umidade")
                export_df = export_df[col_order]

                buffer = BytesIO()
                export_df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.sidebar.download_button(
                    label="Baixar Excel Gerado",
                    data=buffer,
                    file_name="medicoes_sensores.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.sidebar.warning("Carregue os dados dos sensores antes de exportar.")
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Evolução Temporal",
        "Correlação",
        "Gradiente Térmico",
        "Mapa de Calor",
        "Comparação de Sensores",
        "Excursões",
        "Metadados dos Sensores",
        "Métricas Chave",
        "Comparativo Pré × Pós"
    ])
    
    with tab1:
        st.subheader("Evolução da Temperatura ao Longo do Tempo")
        show_external = st.checkbox("Mostrar Temperatura Externa", value=True)

        def render_evolucao(label: str, container, scenario_map, limits=None, gather_only=False):
            scenario = scenario_map.get(label)
            df = scenario["df"]
            sensors = scenario["active_sensors"]
            if df is None or df.empty or df["temp_interna_media"].dropna().empty:
                if gather_only:
                    return {}
                container.info("Sem dados suficientes para este cenário nos filtros atuais.")
                return
            base_cols = ["temp_interna_media", "temp_interna_min", "temp_interna_max"]
            sensor_cols = [s for s in sensors if s in df.columns]
            plot_cols = [c for c in sensor_cols + base_cols if c in df.columns]
            plot_df = df[plot_cols].copy()
            show_ext = show_external and "temp_externa" in df.columns
            if show_ext:
                plot_df["temp_externa"] = df["temp_externa"]
            numeric_df = plot_df.select_dtypes(include=[np.number])
            y_min = float(numeric_df.min().min()) if not numeric_df.empty else None
            y_max = float(numeric_df.max().max()) if not numeric_df.empty else None
            if gather_only:
                return {"y": (y_min, y_max)}
            fig = plot_temperature_over_time(plot_df, sensor_cols, show_ext)
            if limits:
                if limits.get("y"):
                    fig.update_yaxes(range=list(limits["y"]))
            container.plotly_chart(fig, use_container_width=True)
            if "temp_interna_media" in df.columns:
                col_a, col_b, col_c, col_d = container.columns(4)
                col_a.metric("Média", f"{df['temp_interna_media'].mean():.2f}°C")
                col_b.metric("Mínima", f"{df['temp_interna_media'].min():.2f}°C")
                col_c.metric("Máxima", f"{df['temp_interna_media'].max():.2f}°C")
                col_d.metric("Desvio Padrão", f"{df['temp_interna_media'].std():.2f}°C")

        render_scenario_section(dual_labels, render_evolucao, filtered_scenarios, share_limits=True)
    
    with tab2:
        st.subheader("Correlação: Temperatura Externa vs Interna")

        def render_correlacao(label: str, container, scenario_map, limits=None, gather_only=False):
            scenario = scenario_map.get(label)
            df = scenario["df"]
            if df is None or df.empty or "temp_externa" not in df.columns or "temp_interna_media" not in df.columns:
                if gather_only:
                    return {}
                container.warning("Temperatura externa indisponível para este cenário.")
                return
            data = df[["temp_interna_media", "temp_externa"]].dropna()
            if len(data) < 10:
                if gather_only:
                    return {}
                container.info("Dados insuficientes para correlação neste cenário.")
                return
            y_min, y_max = float(data["temp_interna_media"].min()), float(data["temp_interna_media"].max())
            if gather_only:
                return {"y": (y_min, y_max)}
            fig = plot_correlation_scatter(df, "temp_interna_media", "temp_externa")
            if limits:
                if limits.get("y"):
                    fig.update_yaxes(range=list(limits["y"]))
            if fig:
                container.plotly_chart(fig, use_container_width=True)
            corr_pearson, p_pearson = stats.pearsonr(data["temp_externa"], data["temp_interna_media"])
            corr_spearman, p_spearman = stats.spearmanr(data["temp_externa"], data["temp_interna_media"])
            col_a, col_b = container.columns(2)
            col_a.metric("Pearson", f"{corr_pearson:.4f}", delta=f"p = {p_pearson:.2e}")
            col_b.metric("Spearman", f"{corr_spearman:.4f}", delta=f"p = {p_spearman:.2e}")
            if abs(corr_pearson) > 0.7:
                container.info("Correlação forte: interna segue externamente quase em tempo real.")
            elif abs(corr_pearson) > 0.4:
                container.warning("Correlação moderada.")
            else:
                container.success("Correlação fraca: isolamento desacopla bem o ambiente.")

        render_scenario_section(dual_labels, render_correlacao, filtered_scenarios, share_limits=True)
    
    with tab3:
        st.subheader("Gradiente Térmico (Externa - Interna)")

        def render_gradiente(label: str, container, scenario_map, limits=None, gather_only=False):
            scenario = scenario_map.get(label)
            df = scenario["df"]
            if df is None or df.empty or "temp_externa" not in df.columns or "temp_interna_media" not in df.columns:
                if gather_only:
                    return {}
                container.warning("Temperatura externa indisponível para este cenário.")
                return
            data = df[["temp_interna_media", "temp_externa"]].dropna()
            if data.empty:
                if gather_only:
                    return {}
                container.info("Dados insuficientes para calcular gradiente.")
                return
            gradiente = data["temp_externa"] - data["temp_interna_media"]
            y_min, y_max = float(gradiente.min()), float(gradiente.max())
            if gather_only:
                return {"y": (y_min, y_max)}
            fig = plot_thermal_gradient(df, "temp_interna_media", "temp_externa")
            if limits:
                if limits.get("y"):
                    fig.update_yaxes(range=list(limits["y"]))
            container.plotly_chart(fig, use_container_width=True)
            col_a, col_b, col_c = container.columns(3)
            col_a.metric("Gradiente Médio", f"{gradiente.mean():.2f}°C")
            col_b.metric("Mínimo", f"{gradiente.min():.2f}°C")
            col_c.metric("Máximo", f"{gradiente.max():.2f}°C")
            if gradiente.mean() > 5:
                container.success("Excelente isolamento (externa significativamente maior).")
            elif gradiente.mean() > 2:
                container.info("Bom isolamento.")
            elif gradiente.mean() > 0:
                container.warning("Isolamento moderado.")
            else:
                container.error("Isolamento ineficiente (interno mais quente).")

        render_scenario_section(dual_labels, render_gradiente, filtered_scenarios, share_limits=True)
    
    with tab4:
        st.subheader("Mapa de Calor - Temperatura por Hora e Dia")

        def render_heatmap(label: str, container, scenario_map, limits=None, gather_only=False):
            scenario = scenario_map.get(label)
            df = scenario["df"]
            if df is None or df.empty or "temp_interna_media" not in df.columns:
                if gather_only:
                    return {}
                container.warning("Sem temperatura interna disponível.")
                return
            if df["temp_interna_media"].dropna().empty:
                if gather_only:
                    return {}
                container.info("Dados insuficientes para mapa de calor.")
                return
            if gather_only:
                return {}
            fig = plot_heatmap_by_period(df, "temp_interna_media")
            container.plotly_chart(fig, use_container_width=True)

        render_scenario_section(dual_labels, render_heatmap, filtered_scenarios, share_limits=True)
    
    with tab5:
        st.subheader("Comparação entre Sensores")

        def render_sensor_comparison(label: str, container, scenario_map, limits=None, gather_only=False):
            scenario = scenario_map.get(label)
            df = scenario["df"]
            sensors = scenario["active_sensors"]
            if not sensors:
                if gather_only:
                    return {}
                container.info("Nenhum sensor ativo neste cenário com os filtros atuais.")
                return
            plot_df = df[sensors].dropna(how="all")
            if plot_df.empty:
                if gather_only:
                    return {}
                container.info("Dados insuficientes para comparar sensores.")
                return
            y_min = float(plot_df.min().min())
            y_max = float(plot_df.max().max())
            if gather_only:
                return {"y": (y_min, y_max)}
            fig = plot_sensor_comparison(plot_df, sensors)
            if limits and limits.get("y"):
                fig.update_yaxes(range=list(limits["y"]))
            container.plotly_chart(fig, use_container_width=True)
            stats_data = []
            for sensor in sensors:
                values = df[sensor].dropna()
                if len(values) > 0:
                    stats_data.append({
                        "Sensor": sensor,
                        "Média (°C)": f"{values.mean():.2f}",
                        "Mín (°C)": f"{values.min():.2f}",
                        "Máx (°C)": f"{values.max():.2f}",
                        "Desv. Pad. (°C)": f"{values.std():.2f}",
                        "Excursões > limite": int((values > temp_max).sum())
                    })
            if stats_data:
                container.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        render_scenario_section(dual_labels, render_sensor_comparison, filtered_scenarios, share_limits=True)
    
    with tab6:
        st.subheader("Análise de Excursões Acima do Limite")

        def render_excursions(label: str, container, scenario_map, limits=None, gather_only=False):
            scenario = scenario_map.get(label)
            df = scenario["df"]
            sensors = scenario["active_sensors"]
            if not sensors:
                if gather_only:
                    return {}
                container.info("Nenhum sensor ativo neste cenário com os filtros atuais.")
                return
            excursions = []
            for sensor in sensors:
                series = df[[sensor]].dropna()
                if series.empty:
                    continue
                excursions.append(series[sensor])
            if not excursions:
                if gather_only:
                    return {}
                container.info("Dados insuficientes para excursões.")
                return
            y_series = []
            daily_values = []
            for sensor in sensors:
                series = df[[sensor]].dropna()
                if series.empty:
                    continue
                exc = (series[sensor] > temp_max).astype(int)
                grouped = exc.groupby(exc.index.date).sum()
                if not grouped.empty:
                    y_series.append(grouped.max())
                    daily_values.append(grouped)
            if not y_series:
                if gather_only:
                    return {}
                container.info("Dados insuficientes para excursões.")
                return
            y_min = 0
            y_max = float(max(y_series))
            if gather_only:
                return {"y": (y_min, y_max)}
            fig = plot_excursions_over_time(df, sensors, threshold=temp_max)
            if limits:
                if limits.get("y"):
                    fig.update_yaxes(range=list(limits["y"]))
            container.plotly_chart(fig, use_container_width=True)
            exc_rows = []
            for sensor in sensors:
                values = df[sensor].dropna()
                if len(values) == 0:
                    continue
                excursoes = (values > temp_max).sum()
                pct = (excursoes / len(values)) * 100
                exc_rows.append({
                    "Sensor": sensor,
                    "Total de Excursões": excursoes,
                    "% do Tempo": f"{pct:.2f}%",
                    "Temperatura Máxima": f"{values.max():.2f}°C"
                })
            if exc_rows:
                container.dataframe(pd.DataFrame(exc_rows), use_container_width=True)

        render_scenario_section(dual_labels, render_excursions, filtered_scenarios, share_limits=True)
    
    with tab7:
        st.subheader("Informações Detalhadas dos Sensores")
        
        if metadados_sensores:
            for sensor_id, metadata in sorted(metadados_sensores.items()):
                with st.expander(f" {sensor_id} - {metadata.get('modelo', 'N/A')}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Informações do Dispositivo**")
                        if metadata.get("modelo"):
                            st.write(f"**Modelo:** {metadata['modelo']}")
                        if metadata.get("firmware"):
                            st.write(f"**Firmware:** {metadata['firmware']}")
                        if metadata.get("tipo_sensor"):
                            st.write(f"**Tipo:** {metadata['tipo_sensor']}")
                        if metadata.get("numero_viagem"):
                            st.write(f"**Viagem:** {metadata['numero_viagem']}")
                        if metadata.get("qualificacao"):
                            st.write(f"**Qualificação:** {metadata['qualificacao'][:50]}")
                    
                    with col2:
                        st.markdown("**Configuração**")
                        if metadata.get("fuso_horario"):
                            st.write(f"**Fuso Horário:** {metadata['fuso_horario']}")
                        if metadata.get("intervalo_registro"):
                            st.write(f"**Intervalo:** {metadata['intervalo_registro']}")
                        if metadata.get("arquivo_criado"):
                            st.write(f"**Arquivo criado:** {metadata['arquivo_criado']}")
                    
                    # Alarmes
                    if metadata.get("alarmes"):
                        st.markdown("**Limites de Alarme**")
                        for alarme_tipo, alarme_info in metadata["alarmes"].items():
                            st.write(f"- **{alarme_tipo}:** {alarme_info.get('limite', 'N/A')}°C - Status: {alarme_info.get('status', 'N/A')}")
                    
                    # Resumo
                    if metadata.get("resumo"):
                        st.markdown("**Resumo Estatístico**")
                        resumo = metadata["resumo"]
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if "max_temp" in resumo:
                                st.metric("Máximo", f"{resumo['max_temp']:.1f}°C")
                        with col2:
                            if "min_temp" in resumo:
                                st.metric("Mínimo", f"{resumo['min_temp']:.1f}°C")
                        with col3:
                            if "media_temp" in resumo:
                                st.metric("Média", f"{resumo['media_temp']:.1f}°C")
                        with col4:
                            if "mkt" in resumo:
                                st.metric("MKT", f"{resumo['mkt']:.1f}°C")
                        
                        if "leituras_atuais" in resumo:
                            st.write(f"**Total de Leituras:** {resumo['leituras_atuais']:,}")
                        if "primeira_leitura" in resumo:
                            st.write(f"**Primeira Leitura:** {resumo['primeira_leitura']}")
                        if "ultima_leitura" in resumo:
                            st.write(f"**Última Leitura:** {resumo['ultima_leitura']}")
        else:
            st.info("Nenhum metadado extraído dos CSVs.")
    

    with tab8:
        st.subheader("Métricas Chave de Desempenho Térmico")
        metricas = calcular_metricas_energeticas(current_df, temp_max)
        if metricas["delta_t_medio"] is None:
            st.warning("É necessário carregar temperatura interna média e temperatura externa para calcular as métricas.")
        else:
            ajuda_textos = {
                "delta_t_medio": (
                    "Diferença média entre a temperatura interna e a externa. "
                    "Valores positivos indicam que o ambiente interno se mantém mais quente que o exterior; "
                    "quanto menor, melhor o isolamento."
                ),
                "delta_t_p95": (
                    "Valor referente ao percentil 95 da diferença interna-externa. "
                    "Representa os piores 5% das situações de transferência térmica."
                ),
                "graus_hora_acima_limite": (
                    "Grau-hora acima do limite máximo configurado. "
                    "É a soma, ponderada pelo tempo, dos excedentes de temperatura interna acima do limite. "
                    "Serve como estimativa da carga térmica adicional que precisaria ser removida."
                ),
                "slope_temp_ext_int": (
                    "Inclinação da regressão linear entre temperatura externa (x) e interna (y). "
                    "Quanto menor o coeficiente (°C/°C), mais desacoplado o ambiente está das variações externas."
                ),
                "corr_pearson": (
                    "Correlação de Pearson entre temperatura interna e externa. "
                    "Próximo de 1 indica forte dependência; valores próximos de 0 indicam bom isolamento."
                ),
                "lag_horas": (
                    "Defasagem temporal estimada entre oscilações externas e resposta interna. "
                    "Valor positivo: o ambiente interno reage com atraso às mudanças externas. "
                    "Valores baixos sugerem que a temperatura interna segue rapidamente a externa."
                )
            }

            def _fmt(valor: Optional[float], formato: str) -> Optional[str]:
                if valor is None or pd.isna(valor):
                    return None
                return formato.format(valor)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                render_metric_with_help(
                    "ΔT médio (°C)",
                    _fmt(metricas["delta_t_medio"], "{:.2f} °C"),
                    ajuda_textos["delta_t_medio"]
                )
                render_metric_with_help(
                    "ΔT (p95) (°C)",
                    _fmt(metricas["delta_t_p95"], "{:.2f} °C"),
                    ajuda_textos["delta_t_p95"]
                )
            with col_b:
                render_metric_with_help(
                    "Graus-hora acima do limite",
                    _fmt(metricas["graus_hora_acima_limite"], "{:.1f} °C·h"),
                    ajuda_textos["graus_hora_acima_limite"]
                )
                render_metric_with_help(
                    "Inclinação interna vs externa",
                    _fmt(metricas["slope_temp_ext_int"], "{:.2f} °C/°C"),
                    ajuda_textos["slope_temp_ext_int"]
                )
            with col_c:
                render_metric_with_help(
                    "Correlação (Pearson)",
                    _fmt(metricas["corr_pearson"], "{:.3f}"),
                    ajuda_textos["corr_pearson"]
                )
                render_metric_with_help(
                    "Defasagem estimada",
                    _fmt(metricas["lag_horas"], "{:.2f} h"),
                    ajuda_textos["lag_horas"]
                )

            st.caption(
                "As métricas consideram o intervalo filtrado. Para análise comparativa futura, "
                "carregue também os dados dos sensores com a nova tecnologia e utilize os mesmos filtros."
            )

    with tab9:
        st.subheader("Comparativo Pré × Pós")
        if len(comparison_labels) < 2:
            st.info("Selecione pelo menos dois cenários na barra lateral para comparar.")
        else:
            st.markdown(
                """
                **Metodologia normalizada**

                - `ΔT médio` = temperatura interna média menos a externa média (literatura de **cooling degree analysis**).
                - `% tempo > limite (Int)` comparado com `% tempo > limite (Ext)` gera o **Índice de excursões (Int/Ext)**.
                - `Graus-hora normalizado` = graus-hora internos ÷ graus-hora externos (conceito análogo ao *Cooling Degree Hours*).
                - `Fator de amortecimento (Int/Ext)` segue ISO 13786/ASHRAE: razão entre amplitudes interna e externa (quanto menor, melhor barreira térmica).
                - `Índice de estabilidade (σ_int/σ_ext)` indica o quão suavizada está a variabilidade interna em relação ao ambiente externo, usado em estudos de desempenho passivo.
                Valores menores que 1 apontam ambientes mais resilientes mesmo sob condições externas severas.
                """
            )
            comparison_data = []
            for label in comparison_labels:
                scenario_info = filtered_scenarios.get(label)
                if not scenario_info:
                    continue
                summary = scenario_info["summary"]
                comparison_data.append({
                    "Cenário": label,
                    "Período": f"{scenario_info['start'].strftime('%d/%m/%Y')} - {scenario_info['end'].strftime('%d/%m/%Y')}",
                    "Sensores ativos": len(scenario_info["active_sensors"]),
                    "Média interna (°C)": summary["media_interna"],
                    "Min (°C)": summary["min_interna"],
                    "Max (°C)": summary["max_interna"],
                    "ΔT médio (int-ext)": summary["media_delta"],
                    "% tempo > limite (Int)": summary["pct_tempo_acima"],
                    "% tempo > limite (Ext)": summary["pct_ext_acima"],
                    "Índice excursões (Int/Ext)": summary["ratio_excursoes"],
                    "Gradiente médio (°C)": summary["gradiente_medio"],
                    "Correlação externa": summary["corr_pearson"],
                    "Graus-hora acima limite": summary["graus_hora"],
                    "Graus-hora normalizado": summary["graus_hora_norm"],
                "Amplitude interna (°C)": summary["amplitude_interna"],
                "Amplitude externa (°C)": summary["amplitude_externa"],
                "Fator de amortecimento (Int/Ext)": summary["attenuation_factor"],
                "Índice de estabilidade (σ_int/σ_ext)": summary["stability_index"],
                })
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                def style_best(df, better_map):
                    styler = df.style
                    for col, mode in better_map.items():
                        if col not in df.columns:
                            continue
                        values = pd.to_numeric(df[col], errors="coerce")
                        if values.dropna().empty:
                            continue
                        target = values.min() if mode == "min" else values.max()

                        def _style_column(col_series):
                            if col_series.name != col:
                                return [""] * len(col_series)
                            styled = []
                            for val in col_series:
                                numeric = pd.to_numeric(val, errors="coerce")
                                if not pd.isna(numeric) and numeric == target:
                                    styled.append("color: green; font-weight: 600;")
                                else:
                                    styled.append("")
                            return styled

                        styler = styler.apply(_style_column, axis=0)
                    return styler

                quantitative_cols = [
                    "Período",
                    "Sensores ativos",
                    "Média interna (°C)",
                    "Min (°C)",
                    "Max (°C)",
                    "ΔT médio (int-ext)",
                    "% tempo > limite (Int)",
                    "% tempo > limite (Ext)",
                    "Graus-hora acima limite",
                ]
                normalized_cols = [
                    "Gradiente médio (°C)",
                    "Correlação externa",
                    "Índice excursões (Int/Ext)",
                    "Graus-hora normalizado",
                    "Amplitude interna (°C)",
                    "Amplitude externa (°C)",
                    "Fator de amortecimento (Int/Ext)",
                    "Índice de estabilidade (σ_int/σ_ext)",
                ]

                quantitative_map = {
                    "Média interna (°C)": "min",
                    "Min (°C)": "min",
                    "Max (°C)": "min",
                    "ΔT médio (int-ext)": "min",
                    "% tempo > limite (Int)": "min",
                    "% tempo > limite (Ext)": "min",
                    "Graus-hora acima limite": "min",
                }
                normalized_map = {
                    "Gradiente médio (°C)": "max",
                    "Correlação externa": "min",
                    "Índice excursões (Int/Ext)": "min",
                    "Graus-hora normalizado": "min",
                    "Amplitude interna (°C)": "min",
                    "Amplitude externa (°C)": "min",
                    "Fator de amortecimento (Int/Ext)": "min",
                    "Índice de estabilidade (σ_int/σ_ext)": "min",
                }

                st.markdown("#### Indicadores normalizados por temperatura externa")
                st.markdown(
                    "- **Gradiente médio**: externa − interna; positivo indica que o interior permaneceu mais frio.\n"
                    "- **Correlação externa**: quão dependente o interior está da variação externa (menor = melhor isolamento).\n"
                    "- **Índice de excursões (Int/Ext)**: relação entre % do tempo acima do limite interno e externo.\n"
                    "- **Graus-hora normalizado**: carga térmica interna excedente proporcional à carga externa.\n"
                    "- **Amplitude/Fator de amortecimento**: redução das oscilações internas vs. externas (conceito ISO 13786 / ASHRAE).\n"
                    "- **Índice de estabilidade (σ_int/σ_ext)**: quão estável é o ambiente interno comparado ao externo."
                )
                norm_df = comp_df[["Cenário"] + [c for c in normalized_cols if c in comp_df.columns]]
                st.dataframe(
                    style_best(norm_df, normalized_map),
                    use_container_width=True
                )

                st.markdown("#### Indicadores quantitativos (brutos)")
                st.markdown(
                    "- **Média / Min / Max**: estatísticas diretas da temperatura interna.\n"
                    "- **ΔT médio (int-ext)**: diferença média entre interna e externa (negativo indica ambiente mais quente que o exterior).\n"
                    "- **% tempo > limite (Int/Ext)** e **Graus-hora**: tempo e energia excedente acima do limite configurado."
                )
                quant_df = comp_df[["Cenário"] + [c for c in quantitative_cols if c in comp_df.columns]]
                st.dataframe(
                    style_best(quant_df, quantitative_map),
                    use_container_width=True
                )
                # Visual comparativo simples de média e gradiente
                chart_df = comp_df.set_index("Cenário")
                # Separa em 4 gráficos individuais devido às unidades distintas
                graf_cols = st.columns(2)
                with graf_cols[0]:
                    fig_media = go.Figure(go.Bar(
                        x=chart_df.index,
                        y=chart_df["Média interna (°C)"],
                        marker_color="#1f77b4",
                        name="Média interna"
                    ))
                    fig_media.update_layout(title="Média interna (°C)", height=350)
                    st.plotly_chart(fig_media, use_container_width=True)
                with graf_cols[1]:
                    fig_grad = go.Figure(go.Bar(
                        x=chart_df.index,
                        y=chart_df["Gradiente médio (°C)"],
                        marker_color="#ff7f0e",
                        name="Gradiente médio"
                    ))
                    fig_grad.update_layout(title="Gradiente médio (°C)", height=350)
                    st.plotly_chart(fig_grad, use_container_width=True)
                graf_cols2 = st.columns(2)
                with graf_cols2[0]:
                    fig_pct = go.Figure(go.Bar(
                        x=chart_df.index,
                        y=chart_df["% tempo > limite (Int)"],
                        marker_color="#d62728",
                        name="% tempo > limite"
                    ))
                    fig_pct.update_layout(title="% tempo > limite (Int)", height=350)
                    st.plotly_chart(fig_pct, use_container_width=True)
                with graf_cols2[1]:
                    fig_ratio = go.Figure(go.Bar(
                        x=chart_df.index,
                        y=chart_df["Índice excursões (Int/Ext)"],
                        marker_color="#9467bd",
                        name="Índice excursões"
                    ))
                    fig_ratio.update_layout(title="Índice excursões (Int/Ext)", height=350)
                    st.plotly_chart(fig_ratio, use_container_width=True)


    # Rodapé
    st.markdown("---")
    st.markdown("**Dashboard de Análise Térmica 3TC** | Desenvolvido para comparação antes/depois da implementação")

if __name__ == "__main__":
    main()

