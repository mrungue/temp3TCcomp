from pathlib import Path
path = Path.cwd() / 'dashboard_streamlit.py'
text = path.read_text(encoding='utf-8')
replacements = {
    'f"{temp_media:.2f}°C"': 'f"{temp_media:.2f}{CELSIUS}"',
    'f"{filtered_df[\'temp_interna_media\'].mean():.2f}°C"': 'f"{filtered_df[\'temp_interna_media\'].mean():.2f}{CELSIUS}"',
    'f"{filtered_df[\'temp_interna_media\'].min():.2f}°C"': 'f"{filtered_df[\'temp_interna_media\'].min():.2f}{CELSIUS}"',
    'f"{filtered_df[\'temp_interna_media\'].max():.2f}°C"': 'f"{filtered_df[\'temp_interna_media\'].max():.2f}{CELSIUS}"',
    "f\"{filtered_df['temp_interna_media'].std():.2f}°C\"": "f\"{filtered_df['temp_interna_media'].std():.2f}{CELSIUS}\"",
    'f"{gradiente.mean():.2f}°C"': 'f"{gradiente.mean():.2f}{CELSIUS}"',
    'f"{gradiente.min():.2f}°C"': 'f"{gradiente.min():.2f}{CELSIUS}"',
    'f"{gradiente.max():.2f}°C"': 'f"{gradiente.max():.2f}{CELSIUS}"',
    'f"{resumo[\'max_temp\']:.1f}°C"': 'f"{resumo[\'max_temp\']:.1f}{CELSIUS}"',
    'f"{resumo[\'min_temp\']:.1f}°C"': 'f"{resumo[\'min_temp\']:.1f}{CELSIUS}"',
    'f"{resumo[\'media_temp\']:.1f}°C"': 'f"{resumo[\'media_temp\']:.1f}{CELSIUS}"'
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
