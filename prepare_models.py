from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


CV_PATH = "CV.pdf"
FAISS_INDEX_PATH = "app/faiss_db"


loader = PyPDFLoader(CV_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.from_documents(texts, embedding_model)
vectorstore.save_local(FAISS_INDEX_PATH)

print(f"Database created: '{FAISS_INDEX_PATH}'")