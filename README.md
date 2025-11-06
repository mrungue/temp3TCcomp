# Análise de Correlação Térmica - Projeto 3TC

## 📋 Descrição

Sistema de análise de dados de temperatura para comparação de desempenho térmico antes e depois da implementação da tecnologia 3TC em galpão industrial.

O projeto processa dados de múltiplos sensores de temperatura internos e os correlaciona com dados de temperatura externa da cidade, permitindo avaliar a eficácia do isolamento térmico.

---

## 🎯 Objetivos

1. **Correlacionar temperatura interna (galpão) com temperatura externa (cidade)**
2. **Calcular métricas de eficiência do isolamento térmico**
3. **Identificar padrões de comportamento térmico**
4. **Estabelecer baseline para comparação pós-implementação 3TC**
5. **Gerar relatórios profissionais em Excel**

---

## 📁 Estrutura do Projeto

```
.
├── analise_correlacao_termica.py    # Script principal de análise
├── dashboard_streamlit.py            # Dashboard interativo Streamlit
├── gerar_comparativo.py              # Script original (processa ZIP)
├── obter_dados_meteorologicos.py    # Script auxiliar para obter dados externos
├── requirements.txt                  # Dependências Python
├── README.md                         # Este arquivo
├── SUGESTOES_ANALISES.md            # Análises adicionais propostas
└── *.csv                            # Arquivos CSV dos sensores
```

---

## 🚀 Instalação

### 1. Requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

Ou instalar manualmente:
```bash
pip install pandas numpy openpyxl scipy
```

---

## 📊 Uso

### Passo 1: Preparar Dados dos Sensores

Os arquivos CSV dos sensores devem estar no mesmo diretório do script. Os arquivos devem seguir o formato dos sensores RC-51H (EL220500XXXX).

### Passo 2: Obter Dados de Temperatura Externa

Você tem 3 opções:

#### Opção A: Usar dados do INMET (Recomendado para Brasil)
1. Acesse https://portal.inmet.gov.br/
2. Baixe dados históricos da estação meteorológica mais próxima
3. Converta para CSV com colunas: `timestamp`, `temperatura`
4. Salve como `temperatura_externa_cidade.csv`

#### Opção B: Usar API Meteorológica
Execute o script auxiliar:
```bash
python obter_dados_meteorologicos.py
```

Siga as instruções para obter dados via Visual Crossing ou outra API.

#### Opção C: Preencher Template Manualmente
O script pode gerar um template CSV para preenchimento manual.

### Passo 3: Executar Análise Principal

Você tem duas opções:

#### Opção A: Dashboard Interativo (Recomendado)
```bash
streamlit run dashboard_streamlit.py
```

Isso abrirá um dashboard interativo no navegador com visualizações interativas, filtros e gráficos dinâmicos.

#### Opção B: Script de Análise (Gera Excel)
```bash
python analise_correlacao_termica.py
```

### Passo 4: Verificar Resultados

O script gera o arquivo `Analise_Correlacao_Termica_3TC.xlsx` com múltiplas planilhas:

- **dados_consolidados**: Dados brutos alinhados por timestamp
- **correlacao_principal**: Métricas de correlação entre externa e interna
- **gradiente_termico**: Diferença térmica e eficiência do isolamento
- **analise_lag_temporal**: Análise de delay entre mudanças externas/internas
- **analise_por_periodo**: Análise por períodos do dia
- **condicoes_extremas**: Comportamento em dias mais quentes
- **log_importacao**: Log de processamento dos arquivos
- **resumo_estatistico_sensores**: Estatísticas descritivas por sensor

---

## ⚙️ Configuração

Edite as constantes no início do `analise_correlacao_termica.py`:

```python
CSV_DIR = Path(".")  # Diretório com os CSVs
OUTPUT_XLSX = Path("Analise_Correlacao_Termica_3TC.xlsx")
TEMP_MIN, TEMP_MAX = 15.0, 30.0  # Faixa ideal de temperatura
CIDADE = "Cabedelo"  # Nome da cidade
EXTERNAL_TEMP_CSV = Path("temperatura_externa_cidade.csv")  # Arquivo de dados externos
```

---

## 🎨 Dashboard Interativo Streamlit

O dashboard oferece visualizações interativas e análises em tempo real:

### Funcionalidades:
- ✅ **Visualização Temporal**: Gráficos de evolução da temperatura ao longo do tempo
- ✅ **Correlação Externa/Interna**: Scatter plots com regressão linear
- ✅ **Gradiente Térmico**: Análise de eficiência do isolamento
- ✅ **Mapa de Calor**: Visualização por hora e dia da semana
- ✅ **Comparação de Sensores**: Boxplots e estatísticas comparativas
- ✅ **Análise de Excursões**: Identificação de períodos acima do limite
- ✅ **Filtros Interativos**: Por data, sensores e período
- ✅ **Métricas em Tempo Real**: KPIs atualizados conforme filtros

### Como usar:
1. Instale as dependências: `pip install -r requirements.txt`
2. Execute: `streamlit run dashboard_streamlit.py`
3. Acesse no navegador: `http://localhost:8501`
4. Configure diretório dos CSVs e arquivo externo na barra lateral
5. Explore as diferentes abas de análise

---

## 📈 Análises Realizadas

### 1. Correlação Externa vs Interna
- **Correlação de Pearson**: Mede relação linear
- **Correlação de Spearman**: Mede relação monotônica (não linear)
- **Regressão Linear**: Modela relação entre variáveis
- **R²**: Explica variância explicada
- **MAE e RMSE**: Erros de predição

### 2. Gradiente Térmico
- **Diferença Externa - Interna**: Quanto maior, melhor o isolamento
- **Gradiente Percentual**: Redução percentual da temperatura
- **Eficiência do Isolamento**: Classificação (Excelente/Boa/Moderada/Ineficiente)

### 3. Análise de Lag Temporal
- Identifica delay entre mudanças externas e internas
- Encontra lag ótimo para máxima correlação
- Útil para entender inércia térmica

### 4. Análise por Período
- Compara comportamento por horários do dia
- Identifica períodos críticos
- Analisa diferenças dia/noite

### 5. Condições Extremas
- Comportamento em dias mais quentes
- Taxa de excursões acima de 30°C
- Eficiência do isolamento sob stress térmico

---

## 🔄 Workflow de Comparação Antes/Depois

### Fase 1: Baseline (ANTES 3TC)
1. Execute análise com dados atuais
2. Documente métricas principais
3. Salve relatório como "Baseline_Antes_3TC.xlsx"

### Fase 2: Implementação
1. Implemente tecnologia 3TC
2. Aguarde período de estabilização (se necessário)

### Fase 3: Pós-Implementação (DEPOIS 3TC)
1. Colete novos dados dos sensores (mesmo período do ano para comparação justa)
2. Execute análise novamente
3. Compare métricas:
   - Redução de temperatura média
   - Redução de % tempo acima de 30°C
   - Aumento de gradiente térmico
   - Redução de correlação (menor dependência externa)

---

## 📊 Interpretação dos Resultados

### Correlação
- **> 0.7**: Forte correlação (isolamento ineficiente)
- **0.4 - 0.7**: Correlação moderada
- **< 0.4**: Fraca correlação (isolamento eficiente)

### Gradiente Térmico
- **> 5°C**: Excelente isolamento
- **2-5°C**: Bom isolamento
- **0-2°C**: Isolamento moderado
- **< 0°C**: Isolamento ineficiente (interna mais quente que externa)

### Eficiência do Isolamento
Baseado no gradiente térmico:
- **Excelente**: Gradiente > 5°C
- **Boa**: Gradiente 2-5°C
- **Moderada**: Gradiente 0-2°C
- **Ineficiente**: Gradiente < 0°C

---

## 🐛 Solução de Problemas

### Erro: "Nenhum arquivo CSV encontrado"
- Verifique se os arquivos CSV estão no diretório correto
- Verifique se os arquivos têm extensão `.csv`

### Erro: "Dados insuficientes para correlação"
- Verifique se os dados externos cobrem o mesmo período dos internos
- Alinhe os timestamps corretamente

### Erro: Encoding de caracteres
- O script detecta automaticamente encoding (UTF-8, Latin-1, CP1252)
- Se houver problemas, verifique manualmente o encoding dos CSVs

### Dados externos não encontrados
- O script funciona sem dados externos, mas análises de correlação serão limitadas
- Use `obter_dados_meteorologicos.py` para obter dados

---

## 📚 Análises Adicionais Sugeridas

Consulte `SUGESTOES_ANALISES.md` para análises avançadas:

- Análise de inércia térmica
- Mapa de calor temporal
- Análise de variabilidade espacial
- Análise de eficiência energética
- Modelos de previsão (Machine Learning)
- Dashboard interativo

---

## 🤝 Contribuindo

Este é um projeto interno. Para sugestões ou melhorias:

1. Documente a análise proposta
2. Implemente a funcionalidade
3. Teste com dados reais
4. Documente no código

---

## 📝 Licença

Uso interno - Projeto 3TC

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique este README
2. Consulte `SUGESTOES_ANALISES.md`
3. Revise os comentários no código
4. Verifique logs de erro no Excel gerado

---

**Versão**: 1.0  
**Data**: 2024  
**Autor**: Sistema de Análise 3TC

