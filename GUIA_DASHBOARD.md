# 🎨 Guia Rápido - Dashboard Streamlit

## 🚀 Início Rápido

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Executar Dashboard
```bash
streamlit run dashboard_streamlit.py
```

### 3. Acessar
O dashboard abrirá automaticamente no navegador em `http://localhost:8501`

---

## 📊 Funcionalidades do Dashboard

### Barra Lateral (⚙️ Configurações)

1. **Diretório dos CSVs**
   - Caminho para os arquivos CSV dos sensores
   - Padrão: `.` (diretório atual)

2. **Arquivo CSV Temperatura Externa**
   - Caminho para CSV com dados externos (opcional)
   - Formato esperado: colunas `timestamp` e `temperatura`

3. **Temperatura Mínima/Máxima Ideal**
   - Define limites para análise de excursões
   - Padrão: 15°C e 30°C

4. **Filtros**
   - **Período de Análise**: Selecione intervalo de datas
   - **Sensores para Visualizar**: Escolha quais sensores mostrar

---

## 📑 Abas do Dashboard

### 1. 📈 Evolução Temporal
**O que mostra:**
- Gráfico superior: Todas as medições de todos os sensores
- Gráfico inferior: Média, mínima e máxima com faixa
- Opção de mostrar temperatura externa (se disponível)
- Linha de referência no limite de 30°C

**Como usar:**
- Marque/desmarque "Mostrar Temperatura Externa"
- Selecione sensores na barra lateral para focar análise
- Use filtro de data para analisar períodos específicos

**Métricas exibidas:**
- Média, Mínima, Máxima, Desvio Padrão

---

### 2. 🔗 Correlação
**O que mostra:**
- Scatter plot: Temperatura Externa (eixo X) vs Interna (eixo Y)
- Linha de regressão linear
- Linha diagonal (y=x) para referência
- Cores dos pontos representam timestamps

**Métricas:**
- **Correlação de Pearson**: Mede relação linear (-1 a 1)
- **Correlação de Spearman**: Mede relação monotônica
- **Interpretação automática**:
  - 🔴 > 0.7: Correlação forte (isolamento pode melhorar)
  - 🟡 0.4-0.7: Correlação moderada
  - 🟢 < 0.4: Correlação fraca (bom isolamento!)

**Como interpretar:**
- Pontos próximos à linha diagonal: isolamento ineficiente
- Pontos abaixo da diagonal: isolamento funcionando (interna < externa)
- R² alto: boa previsibilidade da interna a partir da externa

---

### 3. 🌡️ Gradiente Térmico
**O que mostra:**
- Gráfico de linha: Gradiente = Externa - Interna ao longo do tempo
- Área preenchida mostra amplitude do gradiente
- Linhas de referência:
  - 0°C: Sem isolamento
  - 2°C: Bom isolamento
  - 5°C: Excelente isolamento

**Métricas:**
- Gradiente Médio, Mínimo, Máximo

**Classificação automática:**
- 🟢 > 5°C: Excelente Isolamento
- 🔵 2-5°C: Bom Isolamento
- 🟡 0-2°C: Isolamento Moderado
- 🔴 < 0°C: Ineficiente (interna mais quente que externa)

**Como interpretar:**
- Valores positivos: isolamento funcionando
- Valores altos: melhor isolamento
- Variações: estabilidade do isolamento

---

### 4. 🗺️ Mapa de Calor
**O que mostra:**
- Heatmap: Temperatura por Hora do Dia (eixo Y) vs Dia (eixo X)
- Cores: Vermelho (quente) → Amarelo → Azul (frio)
- Identifica padrões horários e diários

**Como usar:**
- Identifique horários críticos (mais quentes)
- Compare padrões entre diferentes dias
- Identifique variações sazonais

**Insights:**
- Horários de pico de temperatura
- Padrões de comportamento diurno/noturno
- Identificação de dias anômalos

---

### 5. 📦 Comparação de Sensores
**O que mostra:**
- Boxplot: Distribuição de temperatura por sensor
- Linha de referência no limite (30°C)
- Mostra outliers, quartis, mediana

**Tabela de Estatísticas:**
- Média, Mínima, Máxima por sensor
- Desvio Padrão
- Número de excursões acima do limite

**Como usar:**
- Compare desempenho entre sensores
- Identifique sensores com maior variabilidade
- Identifique pontos problemáticos no galpão

**Insights:**
- Sensores com temperaturas consistentemente mais altas
- Zonas quentes/frias do galpão
- Variabilidade espacial

---

### 6. ⚠️ Excursões
**O que mostra:**
- Gráfico de linha: Minutos acima do limite por dia
- Uma linha por sensor selecionado
- Tabela resumo com estatísticas

**Métricas na Tabela:**
- Total de Excursões por sensor
- % do Tempo acima do limite
- Temperatura Máxima alcançada

**Como usar:**
- Identifique dias com mais problemas
- Compare sensores quanto a excursões
- Avalie eficácia do isolamento em períodos críticos

**Insights:**
- Dias mais problemáticos
- Sensores mais críticos
- Eficácia do isolamento sob stress térmico

---

## 💡 Dicas de Uso

### 1. Análise Comparativa
Para comparar antes/depois da implementação 3TC:
1. Execute análise com dados "antes"
2. Tire screenshots ou exporte gráficos
3. Após implementação, execute novamente
4. Compare métricas lado a lado

### 2. Filtros Estratégicos
- Use filtro de data para períodos específicos
- Compare sensores selecionando apenas alguns
- Analise períodos críticos separadamente

### 3. Interpretação de Correlação
- **Correlação alta (antes 3TC)**: Espera-se redução após implementação
- **Correlação baixa (antes 3TC)**: Já tem bom isolamento, mas pode melhorar ainda mais

### 4. Gradiente Térmico
- **Aumento do gradiente após 3TC**: Sucesso!
- **Gradiente negativo**: Problema crítico (interna mais quente que externa)

### 5. Análise de Excursões
- **Redução de excursões após 3TC**: Objetivo alcançado
- **Foco em sensores com mais excursões**: Priorizar melhorias nesses pontos

---

## 🔧 Solução de Problemas

### Dashboard não abre
- Verifique se Streamlit está instalado: `pip install streamlit`
- Execute: `streamlit --version`
- Tente: `python -m streamlit run dashboard_streamlit.py`

### Erro ao carregar dados
- Verifique se os CSVs estão no diretório correto
- Confirme que os arquivos não estão corrompidos
- Verifique permissões de leitura dos arquivos

### Gráficos não aparecem
- Verifique se Plotly está instalado: `pip install plotly`
- Recarregue a página (F5)
- Verifique console do navegador para erros

### Dados externos não aparecem
- Verifique formato do CSV (colunas: timestamp, temperatura)
- Confirme encoding do arquivo (UTF-8 ou Latin-1)
- Verifique se timestamps estão no mesmo formato dos internos

---

## 📸 Exportação de Dados

O Streamlit permite:
- Screenshots dos gráficos (botão de download)
- Dados filtrados podem ser exportados manualmente
- Use o script `analise_correlacao_termica.py` para gerar Excel completo

---

## 🎯 Próximos Passos

1. **Coletar dados externos** para análise completa
2. **Documentar baseline** (antes 3TC) com screenshots
3. **Após implementação**, executar novamente e comparar
4. **Usar insights** para otimizar pontos problemáticos

---

**Versão**: 1.0  
**Data**: 2024  
**Dashboard**: Streamlit + Plotly

