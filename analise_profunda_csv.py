"""
Análise Profunda e Inteligente de CSVs de Sensores
Extrai e analisa TODAS as informações disponíveis nos CSVs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Optional
from datetime import datetime

def parse_csv_inteligente(filepath: Path) -> Dict:
    """
    Análise profunda do CSV extraindo TODAS as informações disponíveis
    """
    result = {
        "arquivo": filepath.name,
        "metadados": {},
        "configuracao": {},
        "alarmes": {},
        "resumo": {},
        "dados": None,
        "erros": []
    }
    
    # Detecta encoding e delimitador
    try:
        with open(filepath, 'rb') as f:
            sample = f.read(8192)
            for enc in ["latin-1", "cp1252", "iso-8859-1", "utf-8"]:
                try:
                    text = sample.decode(enc, errors="ignore")
                    semi = text.count(";")
                    comma = text.count(",")
                    encoding = enc
                    delimiter = ";" if semi > comma else ","
                    break
                except:
                    continue
    except Exception as e:
        result["erros"].append(f"Erro ao detectar encoding: {e}")
        return result
    
    # Lê o arquivo completo
    try:
        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        result["erros"].append(f"Erro ao ler arquivo: {e}")
        return result
    
    # Parse linha por linha para extrair informações
    linha_atual = 0
    secao_atual = None
    
    # Procura início dos dados
    linha_dados = None
    colunas_dados = None
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Detecta seções
        if "Relatório" in line_clean:
            secao_atual = "header"
        elif "Informação de dispositivo" in line_clean:
            secao_atual = "dispositivo"
        elif "Config. informação" in line_clean or "Config. informa" in line_clean:
            secao_atual = "configuracao"
        elif "Limiar de Alarme" in line_clean or "Limiar de Al" in line_clean:
            secao_atual = "alarmes"
        elif "Resumo" in line_clean:
            secao_atual = "resumo"
        elif "Não." in line_clean and "Tempo" in line_clean:
            # Cabeçalho dos dados
            linha_dados = i
            colunas_dados = [c.strip() for c in line_clean.split(delimiter) if c.strip()]
            continue
        
        # Extrai informações por seção
        if secao_atual == "dispositivo":
            # Modelo do dispositivo
            if "Modelo do dispositivo" in line_clean or "Modelo do dis" in line_clean:
                parts = line_clean.split(delimiter)
                if len(parts) >= 2:
                    result["metadados"]["modelo"] = parts[1].strip()
            
            # Número de série
            if "Número de série" in line_clean or "Número de s" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    if "EL" in p and re.search(r"EL\d+", p):
                        result["metadados"]["numero_serie"] = re.search(r"EL\d+", p).group()
                        break
            
            # Tipo de Sensor
            if "Tipo de Sensor" in line_clean or "Tipo de Se" in line_clean:
                parts = line_clean.split(delimiter)
                for i, p in enumerate(parts):
                    if "Tipo de Sensor" in p or "Tipo de Se" in p:
                        if i + 1 < len(parts):
                            result["metadados"]["tipo_sensor"] = parts[i + 1].strip()
                            break
            
            # Versão do firmware
            if "Versão do firmware" in line_clean or "Versão do f" in line_clean:
                parts = line_clean.split(delimiter)
                for i, p in enumerate(parts):
                    if "firmware" in p.lower():
                        if i + 1 < len(parts):
                            result["metadados"]["firmware"] = parts[i + 1].strip()
                            break
            
            # Número da Viagem
            if "Número da Viagem" in line_clean and "000000" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    if re.match(r"\d{7}", p.strip()):
                        result["metadados"]["numero_viagem"] = p.strip()
            
            if "Qualificacao" in line_clean or "Qualifica" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    if "Qualificacao" in p or "Qualifica" in p:
                        result["metadados"]["qualificacao"] = p.strip()
                        break
        
        elif secao_atual == "configuracao":
            # Intervalo de registro
            if "Intervalo de registro" in line_clean or "Intervalo de r" in line_clean:
                parts = line_clean.split(delimiter)
                for i, p in enumerate(parts):
                    if "Intervalo" in p:
                        if i + 1 < len(parts):
                            result["configuracao"]["intervalo_registro"] = parts[i + 1].strip()
                            break
            
            # Fuso horário
            if "Fuso horário" in line_clean or "Fuso hor" in line_clean:
                parts = line_clean.split(delimiter)
                for i, p in enumerate(parts):
                    if "UTC" in p:
                        result["configuracao"]["fuso_horario"] = p.strip()
                        break
            
            # Modo Iniciar
            if "Modo Iniciar" in line_clean or "Modo Inici" in line_clean:
                parts = line_clean.split(delimiter)
                for i, p in enumerate(parts):
                    if "cronometrado" in p.lower() or "botão" in p.lower():
                        result["configuracao"]["modo_iniciar"] = p.strip()
                        break
        
        elif secao_atual == "alarmes":
            # Limites de alarme
            if "H1:" in line_clean or "H2:" in line_clean or "H3:" in line_clean:
                parts = [p.strip() for p in line_clean.split(delimiter) if p.strip()]
                if len(parts) >= 2:
                    tipo = parts[0].replace(":", "")
                    # Extrai temperatura
                    temp_match = re.search(r"(\d+[.,]\d+)", parts[1])
                    if temp_match:
                        temp_str = temp_match.group(1).replace(",", ".")
                        result["alarmes"][tipo] = {
                            "temperatura_limite": float(temp_str),
                            "duracao_acima": parts[2] if len(parts) > 2 else None,
                            "tempos_acima": parts[3] if len(parts) > 3 else None,
                            "status": parts[4] if len(parts) > 4 else None
                        }
            
            if "L1:" in line_clean or "L2:" in line_clean:
                parts = [p.strip() for p in line_clean.split(delimiter) if p.strip()]
                if len(parts) >= 2:
                    tipo = parts[0].replace(":", "")
                    temp_match = re.search(r"(\d+[.,]\d+)", parts[1])
                    if temp_match:
                        temp_str = temp_match.group(1).replace(",", ".")
                        result["alarmes"][tipo] = {
                            "temperatura_limite": float(temp_str),
                            "duracao_acima": parts[2] if len(parts) > 2 else None,
                            "tempos_acima": parts[3] if len(parts) > 3 else None,
                            "status": parts[4] if len(parts) > 4 else None
                        }
        
        elif secao_atual == "resumo":
            # Máximo
            if "Máximo" in line_clean or "Mximo" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    temp_match = re.search(r"(\d+[.,]\d+)", p)
                    umid_match = re.search(r"(\d+[.,]\d+)%", p)
                    if temp_match:
                        result["resumo"]["maximo_temp"] = float(temp_match.group(1).replace(",", "."))
                    if umid_match:
                        result["resumo"]["maximo_umid"] = float(umid_match.group(1).replace(",", "."))
            
            # Mínimo
            if "Mínimo" in line_clean or "Mnimo" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    temp_match = re.search(r"(\d+[.,]\d+)", p)
                    umid_match = re.search(r"(\d+[.,]\d+)%", p)
                    if temp_match:
                        result["resumo"]["minimo_temp"] = float(temp_match.group(1).replace(",", "."))
                    if umid_match:
                        result["resumo"]["minimo_umid"] = float(umid_match.group(1).replace(",", "."))
            
            # Média
            if "Média" in line_clean or "Mdia" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    temp_match = re.search(r"(\d+[.,]\d+)", p)
                    umid_match = re.search(r"(\d+[.,]\d+)%", p)
                    if temp_match:
                        result["resumo"]["media_temp"] = float(temp_match.group(1).replace(",", "."))
                    if umid_match:
                        result["resumo"]["media_umid"] = float(umid_match.group(1).replace(",", "."))
            
            # MKT (Mean Kinetic Temperature)
            if "MKT" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    temp_match = re.search(r"(\d+[.,]\d+)", p)
                    if temp_match:
                        result["resumo"]["mkt"] = float(temp_match.group(1).replace(",", "."))
            
            # Primeira leitura
            if "Primeira leitura" in line_clean or "Primeira l" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    # Tenta vários formatos de data
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", p)
                    if not date_match:
                        date_match = re.search(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})", p)
                    if date_match:
                        result["resumo"]["primeira_leitura"] = date_match.group(1)
            
            # Última leitura
            if "Ultima leitura" in line_clean or "ltima leitura" in line_clean or "Última leitura" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", p)
                    if not date_match:
                        date_match = re.search(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})", p)
                    if date_match:
                        result["resumo"]["ultima_leitura"] = date_match.group(1)
            
            # Tempo de gravação
            if "Tempo de gravação" in line_clean or "Tempo de g" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    if "D" in p and "H" in p:
                        result["resumo"]["tempo_gravacao"] = p.strip()
            
            # Leituras Atuais
            if "Leituras Atuais" in line_clean or "Leituras A" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    num_match = re.search(r"(\d+)", p)
                    if num_match:
                        result["resumo"]["leituras_atuais"] = int(num_match.group(1))
            
            # Primeiro alarme
            if "Primeiro alarme" in line_clean:
                parts = line_clean.split(delimiter)
                for p in parts:
                    if "Temperatura" in p:
                        date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", p)
                        if not date_match:
                            date_match = re.search(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})", p)
                        if date_match:
                            result["resumo"]["primeiro_alarme_temp"] = date_match.group(1)
    
    # Extrai dados numéricos
    if linha_dados is not None and colunas_dados:
        try:
            # Lê apenas a parte de dados
            df = pd.read_csv(
                filepath,
                sep=delimiter,
                encoding=encoding,
                skiprows=linha_dados,
                engine='python',
                on_bad_lines='skip'
            )
            
            # Limpa colunas
            df.columns = [str(c).strip() for c in df.columns]
            
            # Remove colunas vazias
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Remove linhas completamente vazias
            df = df.dropna(how='all')
            
            result["dados"] = df
            
            # Extrai informações dos dados
            if "Tempo" in df.columns:
                # Parse timestamps
                ts_col = "Tempo"
                ts_raw = df[ts_col].astype(str)
                ts = pd.to_datetime(ts_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=False)
                if ts.notna().sum() < len(ts) * 0.9:
                    ts = pd.to_datetime(ts_raw, format="%d/%m/%Y %H:%M", errors="coerce", utc=False, dayfirst=True)
                df["timestamp"] = ts
                
                # Informações dos dados
                result["resumo"]["dados_inicio"] = str(ts.min()) if ts.notna().any() else None
                result["resumo"]["dados_fim"] = str(ts.max()) if ts.notna().any() else None
                result["resumo"]["total_registros"] = len(df)
            
            if "Temperatura°C" in df.columns or "Temperatura" in df.columns:
                temp_col = "Temperatura°C" if "Temperatura°C" in df.columns else "Temperatura"
                temp = pd.to_numeric(
                    df[temp_col].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                    errors="coerce"
                )
                if temp.notna().any():
                    result["resumo"]["dados_temp_media"] = float(temp.mean())
                    result["resumo"]["dados_temp_min"] = float(temp.min())
                    result["resumo"]["dados_temp_max"] = float(temp.max())
                    result["resumo"]["dados_temp_std"] = float(temp.std())
            
            if "Umidade%RH" in df.columns or "Umidade" in df.columns:
                umid_col = "Umidade%RH" if "Umidade%RH" in df.columns else "Umidade"
                umid = pd.to_numeric(
                    df[umid_col].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\-0-9\.]+)")[0],
                    errors="coerce"
                )
                if umid.notna().any():
                    result["resumo"]["dados_umid_media"] = float(umid.mean())
                    result["resumo"]["dados_umid_min"] = float(umid.min())
                    result["resumo"]["dados_umid_max"] = float(umid.max())
        
        except Exception as e:
            result["erros"].append(f"Erro ao processar dados: {e}")
    
    # Data de criação do arquivo
    for line in lines[:10]:
        if "Arquivo criado em" in line or "Arquivo criado" in line:
            date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
            if date_match:
                result["metadados"]["arquivo_criado_em"] = date_match.group(1)
            break
    
    return result


def analisar_todos_csvs(diretorio: Path) -> List[Dict]:
    """Analisa todos os CSVs do diretório"""
    csv_files = list(diretorio.glob("*.csv"))
    resultados = []
    
    for csv_file in csv_files:
        print(f"Analisando: {csv_file.name}")
        resultado = parse_csv_inteligente(csv_file)
        resultados.append(resultado)
    
    return resultados


if __name__ == "__main__":
    import json
    
    diretorio = Path(".")
    resultados = analisar_todos_csvs(diretorio)
    
    # Salva resultados em JSON
    with open("analise_profunda_resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✓ Análise completa! {len(resultados)} arquivos processados.")
    print(f"✓ Resultados salvos em: analise_profunda_resultados.json")
    
    # Mostra resumo
    print("\n=== RESUMO DA ANÁLISE ===")
    for r in resultados:
        print(f"\nArquivo: {r['arquivo']}")
        if r['metadados'].get('numero_serie'):
            print(f"  Sensor: {r['metadados']['numero_serie']}")
        if r['resumo'].get('leituras_atuais'):
            print(f"  Leituras: {r['resumo']['leituras_atuais']}")
        if r['resumo'].get('media_temp'):
            print(f"  Temp Média (dados): {r['resumo']['dados_temp_media']:.2f}°C")
        if r['alarmes']:
            print(f"  Alarmes configurados: {len(r['alarmes'])}")
        if r['erros']:
            print(f"  ⚠️ Erros: {len(r['erros'])}")

