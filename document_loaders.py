from langchain.document_loaders import WebBaseLoader, NewsURLLoader, SeleniumURLLoader
import nest_asyncio
import time
import re


def loading_webpages(website_urls, loader_class = WebBaseLoader):
    # nest_asyncio.apply()
    loader = loader_class(website_urls)
    # loader.requests_per_second = 1
    docs = loader.load()
    for doc in docs:
        doc.page_content = re.sub('\s+', ' ', doc.page_content).strip()
    return docs


