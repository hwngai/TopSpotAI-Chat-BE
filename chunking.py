from document_loaders import loading_webpages
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(docs, chunk_size=256, chunk_overlap=25):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


# urls = ["https://tamanhhospital.vn/dau-bung/", "https://www.vinmec.com/vi/tieu-hoa-gan-mat/thong-tin-suc-khoe/vi-tri-dau-bung-canh-bao-benh-gi/", "https://hongngochospital.vn/dau-bung-canh-bao-nhieu-benh-ly-nguy-hiem/", "https://www.msdmanuals.com/vi-vn/chuy%C3%AAn-gia/r%E1%BB%91i-lo%E1%BA%A1n-ti%C3%AAu-h%C3%B3a/b%E1%BB%A5ng-c%E1%BA%A5p-v%C3%A0-ph%E1%BA%ABu-thu%E1%BA%ADt-ti%C3%AAu-h%C3%B3a/%C4%91au-b%E1%BB%A5ng-c%E1%BA%A5p-t%C3%ADnh"]
# docs = loading_webpages(website_urls = urls)
# chunks = split_text(docs)
# print(chunks)



