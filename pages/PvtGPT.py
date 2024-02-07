import streamlit as st

from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler

st.set_page_config(page_title="PvtGPT", page_icon="🔐")

st.markdown(
    """
    ## PvtGPT
    """
)

with st.sidebar:
    file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])


@st.cache_resource(show_spinner="Embedding...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/pvt_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/pvt_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=512,
        chunk_overlap=128,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(
        model_name="mistral:latest",
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def display_message_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question only the following documents and not your training data. 
    If you don't know the answer, just say "I don't know". 
    Don't make up an answer.

    Documents: {documents}
    Question: {question}
    """,
)


class ChatCallbackHandler(BaseCallbackHandler):
    msg = ""

    def on_llm_start(self, *args, **kwargs):
        self.msgs = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.msg, "ai")

    def on_llm_new_token(
        self,
        token,
        *args,
        **kwargs,
    ):
        self.msg += token
        self.msgs.markdown(self.msg)


llm = ChatOllama(
    model_name="mistral:latest",
    temperature=0.5,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

if file:
    retriever = embed_file(file)
    send_message(
        "If you have any questions about docs, feel free to ask", "ai", save=False
    )
    display_message_history()
    message = st.chat_input("Ask a question")
    if message:
        send_message(message, "human")
        chain = (
            {
                "documents": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
