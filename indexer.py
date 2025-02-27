# 修改后的indexer.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 使用HuggingFace Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'mps'}
)

# 加载PDF（确保已安装OCR依赖）
loader = PyPDFLoader('LLM.pdf', extract_images=True)
chunks = loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,  # 适当增大chunk size
        chunk_overlap=50
    )
)

# 生成FAISS
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local('LLM.faiss')
print('FAISS saved!')
