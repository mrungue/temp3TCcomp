"""
Script auxiliar para obter dados meteorológicos externos

Este script pode ser adaptado para diferentes fontes de dados:
- INMET (Instituto Nacional de Meteorologia - Brasil)
- OpenWeatherMap API
- Arquivos CSV locais
- Outras APIs meteorológicas

INSTRUÇÕES:
1. Escolha uma fonte de dados
2. Adapte o código conforme necessário
3. Execute para gerar CSV com dados externos
4. Use o CSV gerado no script principal de análise
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests
import json

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
CIDADE = "Cabedelo"
ESTADO = "PB"
LATITUDE = -6.9711  # Ajustar conforme localização exata
LONGITUDE = -34.8378
DATA_INICIO = "2024-02-15"
DATA_FIM = "2024-02-21"
OUTPUT_CSV = Path("temperatura_externa_cidade.csv")

# ============================================================================
# OPÇÃO 1: INMET (Instituto Nacional de Meteorologia - Brasil)
# ============================================================================
# O INMET fornece dados históricos gratuitos via portal
# Acesse: https://portal.inmet.gov.br/
# Ou use a API não oficial (requer pesquisa de estações próximas)

def obter_dados_inmet_manual():
    """
    INSTRUÇÕES MANUAIS PARA INMET:
    
    1. Acesse: https://portal.inmet.gov.br/
    2. Vá em "Dados Históricos"
    3. Selecione a estação meteorológica mais próxima
    4. Baixe os dados para o período desejado
    5. Converta para CSV com colunas: timestamp, temperatura
    6. Use o arquivo gerado no script principal
    """
    print("Para obter dados do INMET:")
    print("1. Acesse https://portal.inmet.gov.br/")
    print("2. Baixe dados históricos da estação mais próxima")
    print("3. Converta para CSV com colunas: timestamp, temperatura")
    return None

# ============================================================================
# OPÇÃO 2: OpenWeatherMap API (requer API key gratuita)
# ============================================================================
def obter_dados_openweather(api_key: str, start_date: str, end_date: str):
    """
    Obtém dados históricos via OpenWeatherMap One Call API
    
    REQUER: API key gratuita (registre-se em openweathermap.org)
    NOTA: A versão gratuita tem limitações para dados históricos
    """
    if not api_key:
        print("ERRO: API key do OpenWeatherMap necessária")
        return None
    
    # OpenWeatherMap One Call API 3.0 (requer assinatura paga para histórico)
    # Alternativa: usar Current Weather API e fazer múltiplas chamadas
    print("⚠ OpenWeatherMap histórico requer assinatura paga")
    print("   Use dados do INMET ou outra fonte gratuita")
    return None

# ============================================================================
# OPÇÃO 3: Visual Crossing Weather API (tem plano gratuito)
# ============================================================================
def obter_dados_visualcrossing(api_key: str, start_date: str, end_date: str,
                               latitude: float, longitude: float):
    """
    Obtém dados históricos via Visual Crossing Weather API
    
    REQUER: API key (registre-se em visualcrossing.com)
    PLANO GRATUITO: 1000 requisições/dia
    """
    if not api_key:
        print("ERRO: API key do Visual Crossing necessária")
        return None
    
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    params = {
        "location": f"{latitude},{longitude}",
        "startDateTime": start_date,
        "endDateTime": end_date,
        "unitGroup": "metric",
        "key": api_key,
        "include": "hours"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Processa dados
        records = []
        for day in data.get("days", []):
            date = day["datetime"]
            for hour in day.get("hours", []):
                timestamp = f"{date} {hour['datetime']}"
                temp = hour.get("temp")
                if temp is not None:
                    records.append({
                        "timestamp": pd.to_datetime(timestamp),
                        "temperatura": temp
                    })
        
        df = pd.DataFrame(records)
        df = df.set_index("timestamp").sort_index()
        df.columns = ["temp_externa"]
        
        return df
        
    except Exception as e:
        print(f"ERRO ao obter dados Visual Crossing: {e}")
        return None

# ============================================================================
# OPÇÃO 4: Template para CSV manual
# ============================================================================
def criar_template_csv():
    """Cria template CSV para preenchimento manual"""
    dates = pd.date_range(start=DATA_INICIO, end=DATA_FIM, freq="1H")
    template = pd.DataFrame({
        "timestamp": dates,
        "temperatura": None  # Preencher manualmente
    })
    template.to_csv("template_temperatura_externa.csv", index=False)
    print("✓ Template criado: template_temperatura_externa.csv")
    print("  Preencha a coluna 'temperatura' com dados da cidade")
    return template

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("OBTENÇÃO DE DADOS METEOROLÓGICOS EXTERNOS")
    print("=" * 70)
    print(f"\nCidade: {CIDADE}, {ESTADO}")
    print(f"Período: {DATA_INICIO} a {DATA_FIM}")
    print(f"Coordenadas: {LATITUDE}, {LONGITUDE}")
    
    print("\nOPÇÕES DISPONÍVEIS:")
    print("1. INMET (manual) - Recomendado para Brasil")
    print("2. Visual Crossing Weather API (requer API key)")
    print("3. Criar template CSV para preenchimento manual")
    
    escolha = input("\nEscolha uma opção (1/2/3): ").strip()
    
    if escolha == "1":
        obter_dados_inmet_manual()
        
    elif escolha == "2":
        api_key = input("Digite sua API key do Visual Crossing: ").strip()
        df = obter_dados_visualcrossing(api_key, DATA_INICIO, DATA_FIM, 
                                       LATITUDE, LONGITUDE)
        if df is not None:
            df.to_csv(OUTPUT_CSV)
            print(f"\n✓ Dados salvos em: {OUTPUT_CSV}")
            print(f"  Total de medições: {len(df)}")
        else:
            print("\n✗ Não foi possível obter os dados")
            
    elif escolha == "3":
        criar_template_csv()
        
    else:
        print("Opção inválida")
    
    print("\n" + "=" * 70)
    print("PRÓXIMOS PASSOS:")
    print("1. Tenha um arquivo CSV com dados de temperatura externa")
    print("2. Configure EXTERNAL_TEMP_CSV no analise_correlacao_termica.py")
    print("3. Execute o script de análise principal")
    print("=" * 70)

if __name__ == "__main__":
    main()

