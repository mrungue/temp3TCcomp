"""
Ferramenta auxiliar para preparar dados pós-instalação a partir de uma
planilha consolidada.

Funcionalidades principais:
- Converter planilhas Excel (uma coluna de timestamp + várias colunas de
  temperatura) em arquivos CSV individuais compatíveis com
  `analise_correlacao_termica.py`.
- (Opcional) Buscar temperaturas externas via Open-Meteo e gerar um
  arquivo CSV alinhado aos timestamps internos para permitir análises de
  correlação completas.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests


def slugify(value: str) -> str:
    """
    Normaliza nomes de sensores para uso em nomes de arquivo.
    Remove acentos, caracteres especiais e converte espaços em "_".
    """
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_only = "".join(c for c in normalized if not unicodedata.combining(c))
    ascii_only = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only)
    ascii_only = ascii_only.strip("_")
    return ascii_only or "sensor"


def detect_timestamp_column(df: pd.DataFrame, preferred: Optional[str]) -> str:
    """Detecta a coluna de timestamp considerando preferências do usuário."""
    if preferred:
        preferred_norm = preferred.strip().lower()
        for col in df.columns:
            if col.strip().lower() == preferred_norm:
                return col

    for col in df.columns:
        norm = col.strip().lower()
        if any(token in norm for token in ("timestamp", "time stamp", "tempo", "time")):
            return col

    raise ValueError(
        "Não foi possível identificar a coluna de timestamp. "
        "Use --timestamp-col para informar o nome exato."
    )


def read_excel(input_path: Path, sheet_name: Optional[str | int]) -> pd.DataFrame:
    """Lê a planilha alvo e remove linhas completamente vazias."""
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("A planilha fornecida não possui dados válidos.")
    return df


def convert_sheet_to_csvs(
    df: pd.DataFrame,
    timestamp_col: str,
    out_dir: Path,
    datetime_format: str,
    sep: str = ";",
) -> List[Path]:
    """Converte cada coluna de temperatura em um CSV individual."""
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    valid_mask = ~timestamps.isna()
    if valid_mask.sum() == 0:
        raise ValueError("A coluna de timestamp não pôde ser convertida para datetime.")

    timestamps = timestamps.loc[valid_mask]
    cleaned = df.loc[valid_mask].reset_index(drop=True)
    created_files: List[Path] = []

    for col in cleaned.columns:
        if col == timestamp_col:
            continue

        series = pd.to_numeric(cleaned[col], errors="coerce")
        sensor_df = pd.DataFrame(
            {
                "Tempo": timestamps.dt.strftime(datetime_format),
                "Temperatura°C": series,
            }
        ).dropna(subset=["Temperatura°C"])

        if sensor_df.empty:
            continue

        filename = f"{slugify(col)}.csv"
        output_path = out_dir / filename
        sensor_df.to_csv(output_path, sep=sep, index=False, encoding="utf-8")
        created_files.append(output_path)

    if not created_files:
        raise ValueError("Nenhuma coluna de temperatura válida foi encontrada.")

    return created_files


def fetch_open_meteo(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "auto",
) -> pd.Series:
    """Baixa série horária de temperatura (°C) da API Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "temperature_unit": "celsius",
        "timezone": timezone,
    }

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30
    )
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly") or {}
    times = hourly.get("time")
    temps = hourly.get("temperature_2m")
    if not times or temps is None:
        raise ValueError("A API Open-Meteo não retornou dados horários válidos.")

    series = pd.Series(
        temps,
        index=pd.to_datetime(times, errors="coerce"),
        name="temp_externa",
        dtype="float64",
    ).dropna()

    if series.empty:
        raise ValueError("A série retornada pela API está vazia.")

    return series.tz_localize(None)


def generate_aligned_external_csv(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    pad_days: int,
    align_freq: str,
    align_tolerance_minutes: int,
    output_path: Path,
) -> Path:
    """Busca dados externos e alinha aos timestamps internos."""
    if timestamps.empty:
        raise ValueError("Lista de timestamps vazia para alinhamento externo.")

    start_date = (timestamps.min().floor("D") - pd.Timedelta(days=pad_days)).date()
    end_date = (timestamps.max().ceil("D") + pd.Timedelta(days=pad_days)).date()
    today = pd.Timestamp.now(tz=None).date()
    if start_date > today:
        start_date = today
    end_date = min(end_date, today)
    if start_date > end_date:
        start_date = end_date

    external_hourly = fetch_open_meteo(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    tolerance = pd.Timedelta(minutes=align_tolerance_minutes)
    resampled = external_hourly.resample(align_freq).interpolate("time")
    aligned = resampled.reindex(timestamps, method="nearest", tolerance=tolerance)
    aligned = aligned.ffill().bfill()

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "temp_externa": aligned.round(3),
        }
    ).dropna(subset=["temp_externa"])

    if df.empty:
        raise ValueError(
            "Não foi possível alinhar dados externos usando a tolerância fornecida."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def build_timestamp_index(sensor_csvs: List[Path], timestamp_col: str) -> pd.DatetimeIndex:
    """Extrai e consolida timestamps a partir de CSVs já convertidos."""
    timestamps: List[pd.Series] = []
    for csv_file in sensor_csvs:
        try:
            df = pd.read_csv(csv_file, sep=None, engine="python")
        except Exception as exc:
            raise ValueError(f"Falha ao ler {csv_file}: {exc}") from exc

        if timestamp_col not in df.columns:
            raise ValueError(f"O arquivo {csv_file} não contém a coluna '{timestamp_col}'.")

        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            continue
        timestamps.append(ts)

    if not timestamps:
        raise ValueError("Não foi possível extrair timestamps dos CSVs gerados.")

    combined = pd.concat(timestamps).drop_duplicates().sort_values()
    return pd.DatetimeIndex(combined)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara dados pós-instalação convertendo planilhas Excel em CSVs "
            "compatíveis com o pipeline e, opcionalmente, gera um CSV de "
            "temperatura externa via Open-Meteo."
        )
    )
    parser.add_argument("--input", required=True, help="Caminho da planilha Excel.")
    parser.add_argument(
        "--sheet",
        default=0,
        help="Nome ou índice da aba a ser utilizada (padrão: 0).",
    )
    parser.add_argument(
        "--timestamp-col",
        default=None,
        help="Nome exato da coluna de timestamp (opcional).",
    )
    parser.add_argument(
        "--datetime-format",
        default="%Y-%m-%d %H:%M:%S",
        help="Formato utilizado ao salvar timestamps nos CSVs.",
    )
    parser.add_argument(
        "--out-sensor-dir",
        required=True,
        help="Diretório de saída para os CSVs de sensores.",
    )
    parser.add_argument(
        "--separator",
        default=";",
        help="Separador utilizado nos CSVs gerados (padrão: ';').",
    )
    parser.add_argument(
        "--fetch-external",
        action="store_true",
        help="Ativa a geração automática de temperatura externa via Open-Meteo.",
    )
    parser.add_argument(
        "--external-output",
        help="Arquivo CSV de saída para temperatura externa (obrigatório com --fetch-external).",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=-6.9711,
        help="Latitude para busca na API Open-Meteo (padrão: Cabedelo/PB).",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=-34.8378,
        help="Longitude para busca na API Open-Meteo (padrão: Cabedelo/PB).",
    )
    parser.add_argument(
        "--pad-days",
        type=int,
        default=1,
        help="Dias extras adicionados antes/depois do período interno para a consulta externa.",
    )
    parser.add_argument(
        "--align-freq",
        default="5min",
        help="Frequência usada para interpolar a série externa antes do alinhamento.",
    )
    parser.add_argument(
        "--align-tolerance-minutes",
        type=int,
        default=45,
        help="Tolerância máxima (minutos) para casar timestamps externos com internos.",
    )

    args = parser.parse_args(argv)

    if args.fetch_external and not args.external_output:
        parser.error("--external-output é obrigatório quando --fetch-external estiver ativo.")

    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    out_sensor_dir = Path(args.out_sensor_dir)

    df = read_excel(input_path, sheet_name=args.sheet)
    timestamp_col = detect_timestamp_column(df, args.timestamp_col)
    sensor_csvs = convert_sheet_to_csvs(
        df=df,
        timestamp_col=timestamp_col,
        out_dir=out_sensor_dir,
        datetime_format=args.datetime_format,
        sep=args.separator,
    )

    print(f"[OK] {len(sensor_csvs)} arquivo(s) de sensor gerado(s) em {out_sensor_dir}")

    if args.fetch_external:
        timestamps = build_timestamp_index(sensor_csvs, timestamp_col="Tempo")
        external_path = Path(args.external_output)
        aligned_path = generate_aligned_external_csv(
            timestamps=timestamps,
            latitude=args.latitude,
            longitude=args.longitude,
            pad_days=args.pad_days,
            align_freq=args.align_freq,
            align_tolerance_minutes=args.align_tolerance_minutes,
            output_path=external_path,
        )
        print(f"[OK] Temperatura externa alinhada salva em {aligned_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)

