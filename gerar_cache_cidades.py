"""
Script para gerar cache de cidades do Open-Meteo
Executa uma vez para baixar e salvar as cidades em CSV
"""

import pandas as pd
import requests
from pathlib import Path
import json

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

def geocode_city(query: str, count: int = 5, country_code: str = None) -> list:
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
            "count": count * 2 if country_code else count,
            "language": "pt",
            "format": "json"
        }
        if country_code:
            params["country_codes"] = country_code
        
        try:
            response = requests.get(OPEN_METEO_GEOCODING_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if results:
                if country_code:
                    results = [r for r in results if r.get('country_code', '').upper() == country_code.upper()]
                if results:
                    return results[:count]
        except Exception:
            continue
    return []

def main():
    print("=" * 70)
    print("GERANDO CACHE DE CIDADES DO OPEN-METEO")
    print("=" * 70)
    
    # Lista de cidades para buscar
    cidades_iniciais = [
        "Cabedelo, PB",
        "João Pessoa, PB",
        "Campina Grande, PB",
        "Recife, PE",
        "Salvador, BA",
        "Fortaleza, CE",
        "São Paulo, SP",
        "Rio de Janeiro, RJ",
        "Brasília, DF",
        "Belo Horizonte, MG",
        "Curitiba, PR",
        "Porto Alegre, RS",
        "Manaus, AM",
        "Belém, PA",
        "Natal, RN"
    ]
    
    todas_cidades = []
    cache_dict = {}
    
    print(f"\nBuscando {len(cidades_iniciais)} cidades na API Open-Meteo...")
    print("(Apenas cidades do Brasil para ser mais rápido)\n")
    
    for i, cidade_query in enumerate(cidades_iniciais, 1):
        print(f"[{i}/{len(cidades_iniciais)}] Buscando: {cidade_query}...", end=" ")
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
                        cache_dict[label] = cidade
                        todas_cidades.append({
                            'label': label,
                            'name': nome,
                            'admin1': cidade.get('admin1', ''),
                            'country': cidade.get('country', ''),
                            'country_code': pais,
                            'latitude': cidade.get('latitude', 0),
                            'longitude': cidade.get('longitude', 0),
                            'timezone': cidade.get('timezone', ''),
                            'elevation': cidade.get('elevation', 0),
                            'dados_completos': json.dumps(cidade)  # Salva dados completos como JSON
                        })
                print(f"✓ {len(results)} cidade(s) encontrada(s)")
            else:
                print("✗ Nenhuma cidade encontrada")
        except Exception as e:
            print(f"✗ Erro: {e}")
    
    # Cria DataFrame e salva em CSV
    if todas_cidades:
        df = pd.DataFrame(todas_cidades)
        df = df.sort_values('label')
        
        output_file = Path("cache_cidades_openmeteo.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n{'=' * 70}")
        print(f"✅ Cache gerado com sucesso!")
        print(f"📁 Arquivo: {output_file}")
        print(f"📊 Total de cidades: {len(df)}")
        print(f"{'=' * 70}")
    else:
        print("\n❌ Nenhuma cidade foi encontrada. Verifique sua conexão com a internet.")

if __name__ == "__main__":
    main()

