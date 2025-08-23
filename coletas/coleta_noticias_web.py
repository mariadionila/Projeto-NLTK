import requests
from bs4 import BeautifulSoup
import pandas as pd 
from newspaper import Article #
from duckduckgo_search import DDGS


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

#CSV
df = pd.DataFrame(resultados)
df.to_csv("../dados/noticias_adultizacao.csv", index=False, encoding="utf-8-sig")

print(resultados)
