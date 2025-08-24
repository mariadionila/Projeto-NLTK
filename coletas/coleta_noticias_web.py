import requests
from bs4 import BeautifulSoup
import pandas as pd 
from newspaper import Article
from duckduckgo_search import DDGS
import re

query = "adultização infantil"
resultados = []

with DDGS() as ddgs:
    for r in ddgs.news(query, max_results=10):  
        titulo = r.get("title")
        descricao = r.get("body")
        url = r.get("url")
        fonte = r.get("source")
        data_pub = r.get("date")
        
        # Agora tenta puxar o texto completo da notícia
        try:
            artigo = Article(url, language="pt")
            artigo.download()
            artigo.parse()
            texto_completo = artigo.text
        except Exception as e:
            texto_completo = "Erro ao extrair conteúdo"

        resultados.append({
            "Título": titulo,
            "Fonte": fonte,
            "Data": data_pub,
            "Descrição": descricao,
            "URL": url,
            "Texto Completo": texto_completo
        })


# Função de limpeza

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"http\S+", "", texto)         # remove links
    texto = re.sub(r"@\w+", "", texto)            # remove menções (@usuario)
    texto = re.sub(r"#\w+", "", texto)            # remove hashtags
    texto = re.sub(r"RT\s+", "", texto)           # remove RT
    texto = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", texto) # remove caracteres especiais/números
    texto = re.sub(r"\s+", " ", texto)            # remove múltiplos espaços
    return texto.lower().strip()


# Criar DataFrame
df = pd.DataFrame(resultados)
df["Texto Limpo"] = df["Texto Completo"].fillna("").astype(str).apply(limpar_texto)


# Salvar em CSV
df.to_csv("../dados/noticias_adultizacao.csv", index=False, encoding="utf-8-sig")
print("✅ Coleta e limpeza concluídas! Salvo em CSV.")
