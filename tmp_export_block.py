from pathlib import Path
text = Path('dashboard_streamlit.py').read_text(encoding='utf-8')
start = text.index('if exportar_excel:')
end = text.index('\n    if internal_df is None or len(sensor_cols) == 0:', start)
print(text[start:end])
