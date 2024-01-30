from llama_index import download_loader
from multiprocessing import Pool
import concurrent.futures
import timeit
from llama_index.node_parser import SentenceSplitter
import chromadb
import uuid


def generate_random_filename():
    random_filename = str(uuid.uuid4())
    return random_filename

def load_contents(urls):
    TrafilaturaWebReader = download_loader('TrafilaturaWebReader')
    trafila_loader = TrafilaturaWebReader()
    content = trafila_loader.load_data(urls)
    content[0].metadata = {"annotationPosition": urls[0]}
    return content


def get_all_contents_multiprocess(urls, pool_size = 16):
    urls_list =[[url] for url in urls]
    with Pool(pool_size) as p:
        results = p.map(load_contents,urls_list)
    return results

def get_all_contents_multithread(urls, max_thread = 30):
    no_thread = min(max_thread, len(urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=no_thread) as executor:
        future = executor.submit(load_contents, urls)
        return future.result()


def get_nodes(contents, chunk_size = 128,chunk_overlap= 0):
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap= chunk_overlap)
    node_list = [node_parser.get_nodes_from_documents(content, show_progress= False) for content in contents]
    nodes = []
    for node_ in node_list:
        for node in node_:
            nodes.append(node)
    return nodes

def embedding(nodes, query, embed_model):
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.text)
        node.embedding = node_embedding
    query_embedding = embed_model.get_text_embedding(query)
    return nodes, query_embedding


async  def chromarank_custom_emb(query, urls, embed_model, chunk_size = 512, chunk_ovelap = 25 ,n_results = 4):
    contents = get_all_contents_multiprocess(urls)
    nodes = get_nodes(contents,chunk_size=chunk_size, chunk_overlap=chunk_ovelap)
    query = query
    nodes, query_embedding = embedding(nodes, query,embed_model)
    chroma_client = chromadb.Client()
    random_name = generate_random_filename()
    collection = chroma_client.create_collection(name=random_name)
    for node in nodes:
        collection.add(
            embeddings=node.embedding,
            documents=node.text,
            ids = node.id_,
            metadatas=node.metadata)
    results = collection.query(
                        query_embeddings= query_embedding,
                        n_results= n_results)
    response = []
    for i in range(0, len(results.get('documents')[0])):
        response.append({
            'pageContent': results.get('documents')[0][i],
            'metadata':  results.get('metadatas')[0][i]
        })
    chroma_client.delete_collection(name=random_name)
    return response