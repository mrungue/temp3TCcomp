# 🔧 Instruções para Executar o Dashboard

## ⚠️ Problema ao Executar em Background

Se o dashboard não iniciou automaticamente, siga estas instruções:

## ✅ Método 1: Executar Manualmente (Recomendado)

### Passo 1: Abrir Terminal/PowerShell
- Pressione `Win + R`
- Digite `powershell` ou `cmd`
- Navegue até o diretório do projeto:
```powershell
cd "C:\Workspace\Pontuais\3TC\Preparação Análise IA - CSV\Preparação Análise IA - CSV"
```

### Passo 2: Executar o Dashboard
```powershell
streamlit run dashboard_streamlit.py --server.port 3333
```

### Passo 3: Acessar no Navegador
O terminal mostrará uma mensagem como:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:3333
```

Abra esse link no navegador.

---

## ✅ Método 2: Usar o Arquivo Batch

1. Clique duas vezes no arquivo `iniciar_dashboard.bat`
2. Aguarde o navegador abrir automaticamente
3. Se não abrir, acesse manualmente: `http://localhost:3333`

---

## ✅ Método 3: Verificar Dependências

Se ainda não funcionar, instale as dependências:

```powershell
pip install -r requirements.txt
```

Ou instale manualmente:
```powershell
pip install streamlit plotly pandas numpy scipy openpyxl
```

---

## 🔍 Verificar se está Rodando

Para verificar se o Streamlit está rodando na porta 3333:

```powershell
netstat -ano | findstr :3333
```

Se aparecer uma linha com `LISTENING`, está rodando!

---

## 🐛 Solução de Problemas

### Erro: "streamlit não é reconhecido"
- Instale o Streamlit: `pip install streamlit`
- Ou use: `python -m streamlit run dashboard_streamlit.py --server.port 3333`

### Erro: "ModuleNotFoundError: No module named 'plotly'"
- Instale: `pip install plotly`

### Erro: "Port already in use"
- Feche outros processos usando a porta 3333
- Ou use outra porta: `--server.port 3334`

### Dashboard não abre no navegador
- Acesse manualmente: `http://localhost:3333`
- Verifique se o firewall não está bloqueando

---

## 📝 Comando Completo

```powershell
cd "C:\Workspace\Pontuais\3TC\Preparação Análise IA - CSV\Preparação Análise IA - CSV"
streamlit run dashboard_streamlit.py --server.port 3333
```

**IMPORTANTE**: Mantenha o terminal aberto enquanto usar o dashboard. Para parar, pressione `Ctrl+C` no terminal.

