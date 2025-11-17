"""
Atalho para executar `analise_correlacao_termica.py` usando os dados
pós-instalação já convertidos e a série externa gerada via Open-Meteo.

Uso:
    set PYTHONIOENCODING=utf-8
    python executar_analise_pos.py
"""

from __future__ import annotations

from pathlib import Path

import analise_correlacao_termica as analise


BASE_DIR = Path(__file__).resolve().parent
POS_DIR = BASE_DIR / "Dados de entrada pós instalação"
CSV_DIR = POS_DIR / "sensores_pos"
EXTERNAL_TEMP_CSV = POS_DIR / "temperatura_externa_openmeteo.csv"
OUTPUT_XLSX = POS_DIR / "Analise_Correlacao_Termica_3TC_pos.xlsx"


def main() -> None:
    analise.CSV_DIR = CSV_DIR
    analise.EXTERNAL_TEMP_CSV = EXTERNAL_TEMP_CSV
    analise.OUTPUT_XLSX = OUTPUT_XLSX
    analise.main()


if __name__ == "__main__":
    main()

