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
from typing import Dict, List, Optional
from io import BytesIO
import requests
import warnings
warnings.filterwarnings('ignore')

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
    
    # Sidebar - Configurações
    st.sidebar.header("Configurações")
    
    # Diretório fixo dos CSVs
    csv_dir = "Dados de entrada pré instalação"
    
    fonte_externa = st.sidebar.selectbox(
        "Fonte dos dados de temperatura externa",
        ["Nenhum", "Arquivo CSV", "API Open-Meteo"],
        index=2
    )
    
    external_csv = ""
    selected_location: Optional[Dict] = None
    
    if fonte_externa == "Arquivo CSV":
        external_csv = st.sidebar.text_input(
            "Arquivo CSV Temperatura Externa",
            value="",
            help="Informe um arquivo com timestamp e temperatura externa."
        )
    elif fonte_externa == "API Open-Meteo":
        # Inicializa estados
        if 'selected_city' not in st.session_state:
            st.session_state['selected_city'] = None
        
        selected_location = None
        
        # Arquivo onde as cidades são salvas
        cidades_file = Path("cidades_openmeteo.csv")
        
        # Função para buscar cidades na API e salvar no arquivo
        def _buscar_e_salvar_cidades():
            """Busca cidades na API e salva no arquivo CSV"""
            cidades_iniciais = [
                "Cabedelo, PB", "João Pessoa, PB", "Campina Grande, PB",
                "Recife, PE", "Salvador, BA", "Fortaleza, CE",
                "São Paulo, SP", "Rio de Janeiro, RJ", "Brasília, DF",
                "Belo Horizonte, MG", "Curitiba, PR", "Porto Alegre, RS",
                "Manaus, AM", "Belém, PA", "Natal, RN"
            ]
            
            todas_cidades = []
            cache_dict = {}
            
            for cidade_query in cidades_iniciais:
                try:
                    results = geocode_city(cidade_query, count=3, country_code="BR")
                    if results:
                        for cidade in results:
                            nome = cidade.get('name', '')
                            admin = cidade.get('admin1') or cidade.get('country', '')
                            pais = cidade.get('country_code', '')
                            label = f"{nome}"
                            if admin:
                                label += f", {admin}"
                            if pais:
                                label += f" ({pais.upper()})"
                            
                            # Evita duplicatas
                            if label not in cache_dict:
                                cache_dict[label] = True
                                todas_cidades.append({
                                    'label': label,
                                    'name': nome,
                                    'admin1': cidade.get('admin1', ''),
                                    'country': cidade.get('country', ''),
                                    'country_code': pais,
                                    'latitude': cidade.get('latitude', 0),
                                    'longitude': cidade.get('longitude', 0),
                                    'timezone': cidade.get('timezone', ''),
                                    'elevation': cidade.get('elevation', 0)
                                })
                except Exception:
                    continue
            
            # Salva no arquivo CSV
            if todas_cidades:
                df = pd.DataFrame(todas_cidades)
                df = df.sort_values('label')
                df.to_csv(cidades_file, index=False, encoding='utf-8-sig')
                return df
            return None
        
        # Carrega cidades do arquivo (ou busca e salva se não existir)
        if 'cities_loaded' not in st.session_state:
            with st.sidebar:
                with st.spinner("🔍 Carregando cidades..."):
                    if cidades_file.exists():
                        try:
                            df_cidades = pd.read_csv(cidades_file, encoding='utf-8-sig')
                        except Exception as e:
                            st.warning(f"Erro ao ler arquivo de cidades: {e}. Buscando na API...")
                            df_cidades = _buscar_e_salvar_cidades()
                    else:
                        st.info("📥 Arquivo de cidades não encontrado. Buscando na API...")
                        df_cidades = _buscar_e_salvar_cidades()
                    
                    if df_cidades is not None and len(df_cidades) > 0:
                        # Prepara dicionário de cidades
                        cities_dict = {}
                        cities_options = []
                        
                        for _, row in df_cidades.iterrows():
                            label = row['label']
                            cidade = {
                                'name': row['name'],
                                'admin1': row.get('admin1', ''),
                                'country': row.get('country', ''),
                                'country_code': row.get('country_code', 'BR'),
                                'latitude': float(row.get('latitude', 0)),
                                'longitude': float(row.get('longitude', 0)),
                                'timezone': row.get('timezone', ''),
                                'elevation': float(row.get('elevation', 0))
                            }
                            cities_dict[label] = cidade
                            cities_options.append(label)
                        
                        # Salva no session_state
                        st.session_state['cities_dict'] = cities_dict
                        st.session_state['cities_options'] = cities_options
                        
                        # Define Cabedelo como cidade padrão selecionada
                        if not st.session_state.get('selected_city'):
                            for label, cidade in cities_dict.items():
                                nome = cidade.get('name', '').lower()
                                admin = cidade.get('admin1', '').lower()
                                if 'cabedelo' in nome and ('paraíba' in admin or 'paraiba' in admin):
                                    st.session_state['selected_city'] = cidade
                                    break
                        
                        st.session_state['cities_loaded'] = True
                    else:
                        st.error("Não foi possível carregar as cidades.")
                        st.session_state['cities_loaded'] = True
        
        # Prepara lista de opções para o selectbox
        opcoes_cidades = st.session_state.get('cities_options', []).copy()
        
        # Garante que há pelo menos uma opção
        if not opcoes_cidades:
            opcoes_cidades = ["Carregando..."]
        
        # Determina índice padrão (cidade selecionada)
        default_idx = 0
        if st.session_state.get('selected_city'):
            selected = st.session_state['selected_city']
            nome = selected.get('name', '')
            admin = selected.get('admin1') or selected.get('country', '')
            pais = selected.get('country_code', '')
            label_atual = f"{nome}"
            if admin:
                label_atual += f", {admin}"
            if pais:
                label_atual += f" ({pais.upper()})"
            
            if label_atual in opcoes_cidades:
                default_idx = opcoes_cidades.index(label_atual)
        
        # SELECTBOX ÚNICO - o Streamlit permite digitar para filtrar opções existentes
        cidade_selecionada = st.sidebar.selectbox(
            "🌍 Cidade",
            options=opcoes_cidades,
            index=default_idx if opcoes_cidades else 0,
            key="cidade_selectbox_final",
            help="Digite para filtrar ou selecione uma cidade. Cabedelo já está carregado."
        )
        
        # Atualiza cidade selecionada
        cities_dict = st.session_state.get('cities_dict', {})
        if cidade_selecionada and cidade_selecionada in cities_dict:
            selected_location = cities_dict[cidade_selecionada]
            st.session_state['selected_city'] = selected_location
        
        # Garante selected_location
        if st.session_state.get('selected_city') and not selected_location:
            selected_location = st.session_state['selected_city']
    
    temp_min = st.sidebar.number_input("Temperatura mínima ideal (°C)", value=15.0)
    temp_max = st.sidebar.number_input("Temperatura máxima ideal (°C)", value=30.0)
    
    # Carrega dados
    with st.spinner("Carregando dados..."):
        internal_df, sensor_cols, metadados_sensores = load_internal_data(Path(csv_dir))
        
        external_df = None
        external_success = None
        external_error = None
        
        if internal_df is not None and len(internal_df):
            if fonte_externa == "Arquivo CSV" and external_csv:
                try:
                    external_df = load_external_data(Path(external_csv))
                    if external_df is None or external_df.empty:
                        external_error = "Não foi possível interpretar o CSV de temperatura externa."
                    else:
                        external_success = f"Temperatura externa carregada de {external_csv}."
                except Exception as exc:
                    external_error = f"Erro ao ler CSV externo: {exc}"
                    external_df = None
            elif fonte_externa == "API Open-Meteo":
                if not selected_location:
                    external_error = "⚠️ Selecione uma cidade para buscar dados de temperatura externa."
                else:
                    try:
                        start_ts = internal_df.index.min()
                        end_ts = internal_df.index.max()
                        
                        # Valida coordenadas
                        lat = selected_location.get("latitude")
                        lon = selected_location.get("longitude")
                        if lat is None or lon is None:
                            external_error = "Coordenadas inválidas para a cidade selecionada."
                        else:
                            with st.spinner(f"🌡️ Buscando dados de temperatura para {selected_location.get('name')}..."):
                                external_df = fetch_external_temperature_from_api(
                                    lat,
                                    lon,
                                    start_ts,
                                    end_ts,
                                    selected_location.get("timezone")
                                )
                            
                            if external_df is None or external_df.empty:
                                external_error = (
                                    f"API não retornou dados de temperatura para {selected_location.get('name')} "
                                    f"no período de {start_ts.date()} a {end_ts.date()}. "
                                    f"Verifique se as datas estão dentro do período disponível na API."
                                )
                            else:
                                external_success = (
                                    f"✅ Temperatura externa obtida via Open-Meteo para {selected_location.get('name')} "
                                    f"({selected_location.get('country_code')}). "
                                    f"Total de {len(external_df)} medições carregadas."
                                )
                    except Exception as exc:
                        external_error = f"❌ Falha ao consultar API Open-Meteo: {str(exc)}"
                        external_df = None
            
            if external_df is not None and not external_df.empty:
                internal_df = internal_df.join(external_df, how="outer")
                if "temp_externa" in internal_df.columns:
                    internal_df["temp_externa"] = pd.to_numeric(
                        internal_df["temp_externa"],
                        errors="coerce"
                    )
                    internal_df["temp_externa"] = (
                        internal_df["temp_externa"]
                        .interpolate(method="time")
                        .ffill()
                        .bfill()
                    )
                    internal_df = internal_df.sort_index()
        
        # Exibe mensagens de sucesso ou erro
        if external_success:
            st.sidebar.success(external_success)
        if external_error:
            st.sidebar.error(f"⚠️ {external_error}")
    
    # Variável de exportação será definida depois dos filtros

    if internal_df is None or len(sensor_cols) == 0:
        st.error("Nenhum dado de sensor encontrado. Verifique se há arquivos CSV na pasta 'Dados de entrada pré instalação'.")
        return
    
    # Informações gerais
    st.header("Visão Geral")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Número de Sensores", len(sensor_cols))
    
    with col2:
        st.metric("Total de Medições", f"{len(internal_df):,}")
    
    with col3:
        if "temp_interna_media" in internal_df.columns:
            temp_media = internal_df["temp_interna_media"].mean()
            st.metric("Temperatura Média Interna", f"{temp_media:.2f}°C")
    
    with col4:
        if "temp_interna_media" in internal_df.columns:
            excursoes = (internal_df["temp_interna_media"] > temp_max).sum()
            st.metric("Excursões Acima do Limite", f"{excursoes}")
    
    # Filtros de data
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros")
    
    min_date = internal_df.index.min().date()
    max_date = internal_df.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Período de Análise",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_df = internal_df[
            (internal_df.index.date >= date_range[0]) & 
            (internal_df.index.date <= date_range[1])
        ].copy()
    else:
        filtered_df = internal_df.copy()
    
    # Seleção de sensores
    selected_sensors = st.sidebar.multiselect(
        "Sensores para análise",
        options=sensor_cols,
        default=sensor_cols
    )
    
    # Recalcula valores médios usando apenas os sensores selecionados
    if selected_sensors:
        filtered_df["temp_interna_media"] = filtered_df[selected_sensors].mean(axis=1)
        filtered_df["temp_interna_min"] = filtered_df[selected_sensors].min(axis=1)
        filtered_df["temp_interna_max"] = filtered_df[selected_sensors].max(axis=1)
        filtered_df["temp_interna_std"] = filtered_df[selected_sensors].std(axis=1)
    else:
        # Se nenhum sensor selecionado, mantém valores originais ou define como NaN
        filtered_df["temp_interna_media"] = np.nan
        filtered_df["temp_interna_min"] = np.nan
        filtered_df["temp_interna_max"] = np.nan
        filtered_df["temp_interna_std"] = np.nan
    
    # Botão de exportar (último do menu)
    st.sidebar.markdown("---")
    exportar_excel = st.sidebar.button("📤 Exportar medições (Excel)", use_container_width=True)
    
    # Lógica de exportação (executada DEPOIS do botão ser criado)
    if exportar_excel:
        if internal_df is not None and sensor_cols:
            hum_map = {
                sensor: f"{sensor}_umidade"
                for sensor in sensor_cols
                if f"{sensor}_umidade" in internal_df.columns
            }
            temp_long = (
                internal_df[sensor_cols]
                .reset_index()
                .rename(columns={"index": "timestamp"})
                .melt(id_vars="timestamp", var_name="sensor", value_name="temperatura")
                .dropna(subset=["temperatura"])
            )
            if temp_long.empty:
                st.sidebar.warning("Não há medições válidas para exportar.")
            else:
                # Adiciona temperatura externa se disponível
                if "temp_externa" in internal_df.columns:
                    temp_externa_df = (
                        internal_df[["temp_externa"]]
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
                        internal_df[hum_cols]
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Evolução Temporal",
        "Correlação",
        "Gradiente Térmico",
        "Mapa de Calor",
        "Comparação de Sensores",
        "Excursões",
        "Metadados dos Sensores",
        "Métricas Chave"
    ])
    
    with tab1:
        st.subheader("Evolução da Temperatura ao Longo do Tempo")
        show_external = st.checkbox("Mostrar Temperatura Externa", value=True)
        
        # Filtra sensores selecionados
        if selected_sensors:
            plot_df = filtered_df[selected_sensors + ["temp_interna_media", "temp_interna_min", "temp_interna_max"]]
            if "temp_externa" in filtered_df.columns and show_external:
                plot_df["temp_externa"] = filtered_df["temp_externa"]
        else:
            plot_df = filtered_df[["temp_interna_media", "temp_interna_min", "temp_interna_max"]]
            if "temp_externa" in filtered_df.columns and show_external:
                plot_df["temp_externa"] = filtered_df["temp_externa"]
        
        fig = plot_temperature_over_time(plot_df, selected_sensors if selected_sensors else [], show_external)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas resumidas
        if "temp_interna_media" in filtered_df.columns:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Média", f"{filtered_df['temp_interna_media'].mean():.2f}°C")
            with col2:
                st.metric("Mínima", f"{filtered_df['temp_interna_media'].min():.2f}°C")
            with col3:
                st.metric("Máxima", f"{filtered_df['temp_interna_media'].max():.2f}°C")
            with col4:
                st.metric("Desvio Padrão", f"{filtered_df['temp_interna_media'].std():.2f}°C")
    
    with tab2:
        st.subheader("Correlação: Temperatura Externa vs Interna")
        
        if "temp_externa" in filtered_df.columns and "temp_interna_media" in filtered_df.columns:
            fig = plot_correlation_scatter(filtered_df, "temp_interna_media", "temp_externa")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Métricas de correlação
                data = filtered_df[["temp_interna_media", "temp_externa"]].dropna()
                if len(data) > 10:
                    corr_pearson, p_pearson = stats.pearsonr(data["temp_externa"], data["temp_interna_media"])
                    corr_spearman, p_spearman = stats.spearmanr(data["temp_externa"], data["temp_interna_media"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Correlação de Pearson", f"{corr_pearson:.4f}", 
                                 delta=f"p = {p_pearson:.2e}")
                    with col2:
                        st.metric("Correlação de Spearman", f"{corr_spearman:.4f}",
                                 delta=f"p = {p_spearman:.2e}")
                    
                    # Interpretação
                    if abs(corr_pearson) > 0.7:
                        st.info("**Correlação Forte**: A temperatura interna está fortemente dependente da externa. O isolamento pode ser melhorado.")
                    elif abs(corr_pearson) > 0.4:
                        st.warning("**Correlação Moderada**: A temperatura interna tem dependência moderada da externa.")
                    else:
                        st.success("**Correlação Fraca**: A temperatura interna é pouco dependente da externa. Bom isolamento!")
        else:
            st.warning("âš ï¸ Dados de temperatura externa não disponíveis. Carregue um arquivo CSV com dados externos.")
    
    with tab3:
        st.subheader("Gradiente Térmico (Externa - Interna)")
        
        if "temp_externa" in filtered_df.columns and "temp_interna_media" in filtered_df.columns:
            fig = plot_thermal_gradient(filtered_df, "temp_interna_media", "temp_externa")
            st.plotly_chart(fig, use_container_width=True)
            
            # Estatísticas do gradiente
            data = filtered_df[["temp_interna_media", "temp_externa"]].dropna()
            gradiente = data["temp_externa"] - data["temp_interna_media"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gradiente Médio", f"{gradiente.mean():.2f}°C")
            with col2:
                st.metric("Gradiente Mínimo", f"{gradiente.min():.2f}°C")
            with col3:
                st.metric("Gradiente Máximo", f"{gradiente.max():.2f}°C")
            
            # Classificação
            if gradiente.mean() > 5:
                st.success("**Excelente Isolamento**: Gradiente médio > 5°C")
            elif gradiente.mean() > 2:
                st.info("**Bom Isolamento**: Gradiente médio entre 2-5°C")
            elif gradiente.mean() > 0:
                st.warning("**Isolamento Moderado**: Gradiente médio entre 0-2°C")
            else:
                st.error("**Isolamento Ineficiente**: Gradiente negativo (interna mais quente que externa)")
        else:
            st.warning("Dados de temperatura externa não disponíveis.")
    
    with tab4:
        st.subheader("Mapa de Calor - Temperatura por Hora e Dia")
        if "temp_interna_media" in filtered_df.columns:
            fig = plot_heatmap_by_period(filtered_df, "temp_interna_media")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados insuficientes para mapa de calor.")
    
    with tab5:
        st.subheader("Comparação entre Sensores")
        if selected_sensors:
            plot_df = filtered_df[selected_sensors]
            fig = plot_sensor_comparison(plot_df, selected_sensors)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de estatísticas
            st.subheader("Estatísticas por Sensor")
            stats_data = []
            for sensor in selected_sensors:
                values = filtered_df[sensor].dropna()
                if len(values) > 0:
                    stats_data.append({
                        "Sensor": sensor,
                        "Média (°C)": f"{values.mean():.2f}",
                        "Mín (°C)": f"{values.min():.2f}",
                        "Máx (°C)": f"{values.max():.2f}",
                        "Desv. Pad. (°C)": f"{values.std():.2f}",
                        "Excursões > 30°C": int((values > temp_max).sum())
                    })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        else:
            st.info("Selecione sensores na barra lateral para visualizar.")
    
    with tab6:
        st.subheader("Análise de Excursões Acima do Limite")
        if selected_sensors:
            fig = plot_excursions_over_time(filtered_df, selected_sensors, threshold=temp_max)
            st.plotly_chart(fig, use_container_width=True)
            
            # Resumo de excursões
            st.subheader("Resumo de Excursões")
            exc_data = []
            for sensor in selected_sensors:
                values = filtered_df[sensor].dropna()
                if len(values) > 0:
                    excursoes = (values > temp_max).sum()
                    pct = (excursoes / len(values)) * 100
                    exc_data.append({
                        "Sensor": sensor,
                        "Total de Excursões": excursoes,
                        "% do Tempo": f"{pct:.2f}%",
                        "Temperatura Máxima": f"{values.max():.2f}°C"
                    })
            
            if exc_data:
                st.dataframe(pd.DataFrame(exc_data), use_container_width=True)
        else:
            st.info("Selecione sensores na barra lateral para visualizar.")
    
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
        metricas = calcular_metricas_energeticas(filtered_df, temp_max)
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


    # Rodapé
    st.markdown("---")
    st.markdown("**Dashboard de Análise Térmica 3TC** | Desenvolvido para comparação antes/depois da implementação")

if __name__ == "__main__":
    main()

