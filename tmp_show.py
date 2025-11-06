from pathlib import Path
text = Path('dashboard_streamlit.py').read_text(encoding='utf-8')
lines = text.splitlines()
for i in range(620, 780):
    line = lines[i]
    print(f"{i+1}: {line.encode('unicode_escape').decode()}")
