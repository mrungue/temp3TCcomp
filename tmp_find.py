from pathlib import Path
text = Path('dashboard_streamlit.py').read_text(encoding='utf-8')
start = text.index('sensor_cols = [c for c in wide.columns')
print(text[start:start+120])
