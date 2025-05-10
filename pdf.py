import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader 



Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index...", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))

    return index


pdf_path = os.path.join("data", "Vietnam.pdf")
vietnam_pdf = PDFReader().load_data(file=pdf_path)
vietnam_index = get_index(vietnam_pdf, "vietnam")
vietname_engine = vietnam_index.as_query_engine()

