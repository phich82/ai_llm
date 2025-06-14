# Retrieval Augmented Generation = RAG

import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
from openai import OpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from enum import Enum, unique



@unique
class ProviderEnum(Enum):
    CHROMA = 'Chroma'
    FAISS = 'Faiss'
    OPENAI = 'OpenAI'
    HUGGING_FACE = 'Hugging Face'

    @classmethod
    def from_name(cls, name: str):
        for (service_name, service) in cls._member_map_.items():
            if service_name == name:
                return service
        raise Exception(f'Name not found: {name}')

    @classmethod
    def get_names(cls):
        return cls._member_names_


load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

db_name = "vector_db"
OPENAI_MODEL = "gpt-4.1"
openai = OpenAI()

context = {}

def load_files(glob_path: str='', filename_separator: str=' ', context: dict={}):
    for filename in glob.glob(glob_path):
        name = filename.split(filename_separator)[-1][:-3]
        doc = ""
        with open(filename, "r", encoding="utf-8") as f:
            doc = f.read()
        context[name] = doc
    return context

# Employees
load_files(glob_path="knowledge-base/employees/*", context=context)
# Products
load_files(glob_path="knowledge-base/products/*", context=context, filename_separator=os.sep)

print(context.keys())


def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context

def add_context(message: str):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message

def chat(message, history):
    system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."
    messages = [{"role": "system", "content": system_message}] + history
    messages.append({"role": "user", "content": add_context(message)})

    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response


class RAG:

    db_name: str = "vector_db"
    OPENAI_MODEL: str = "gpt-4.1"
    HUGGING_FACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_INDEX_NAME: str = 'indexing'
    COLLECTION_NAME: str = 'langchain'
    DEBUG: bool = True

    vectorstore: Chroma | FAISS
    doc_types: set
    openai: OpenAI = OpenAI()
    conversation_chain: ConversationalRetrievalChain = None
    db_provider: ProviderEnum = ProviderEnum.CHROMA
    embedding_provider: ProviderEnum = ProviderEnum.HUGGING_FACE

    def __init__(self,
                 folder: str='knowledge-base/*',
                 db_provider: ProviderEnum=ProviderEnum.CHROMA,
                 embedding_provider: ProviderEnum=ProviderEnum.HUGGING_FACE):
        if db_provider is None or db_provider != '':
            self.db_provider = db_provider

        if embedding_provider is None or embedding_provider != '':
            self.embedding_provider = embedding_provider

        self.store_embeddings(folder=folder, db_provider=self.db_provider)

        if self.conversation_chain is None:
            self.conversation_chain = self.get_conversation_chain()

    def get_conversation_chain(self, temperature: float=0.7, memory_key='chat_history', return_messages: bool=True):
        # create a new Chat with OpenAI
        llm = ChatOpenAI(temperature=temperature, model_name=self.OPENAI_MODEL)

        # set up the conversation memory for the chat
        memory = ConversationBufferMemory(memory_key=memory_key, return_messages=return_messages)

        # the retriever is an abstraction over the VectorStore that will be used during RAG; k is how many chunks to use
        # retriever = self.vectorstore.as_retriever()
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 25})

        # putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
        callbacks = [StdOutCallbackHandler()] if self.DEBUG else []
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            callbacks=callbacks)

        return conversation_chain

    def load_documents(self, folder: str='knowledge-base/*') -> list:
        folders = glob.glob(folder)
        text_loader_kwargs = {'encoding': 'utf-8'} # {'autodetect_encoding': True}
        documents = []
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        return documents

    def store_embeddings(self, folder: str='knowledge-base/*', db_provider: ProviderEnum=ProviderEnum.CHROMA):
        # Load data
        documents = self.load_documents(folder)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Doc types
        self.doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
        print(f"Document types found: {', '.join(self.doc_types)}")

        # Embedding
        if self.embedding_provider == ProviderEnum.OPENAI:
            print('embeddings ===> OpenAIEmbeddings')
            # Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
            embeddings = OpenAIEmbeddings()
        else:
            print('embeddings ===> HuggingFaceEmbeddings')
            # If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers
            embeddings = HuggingFaceEmbeddings(model_name=self.HUGGING_FACE_EMBEDDING_MODEL)

        # Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch
        if os.path.exists(db_name):
            if self.db_provider == ProviderEnum.CHROMA:
                print('Deleting the old vector store')
                Chroma(persist_directory=db_name, embedding_function=embeddings, collection_name=self.COLLECTION_NAME).delete_collection()
                print('Deleted the old vector store')
            elif self.db_provider == ProviderEnum.FAISS:
                print('Deleting the old vector store')
                # vectorstore = FAISS.load_local(folder_path=db_name, embeddings=embeddings, allow_dangerous_deserialization=True)
                # vectorstore.delete([key for key, doc in vectorstore.docstore._dict.items()])
                os.remove(f'{db_name}/{self.EMBEDDING_INDEX_NAME}.faiss')
                os.remove(f'{db_name}/{self.EMBEDDING_INDEX_NAME}.pkl')
                print('Deleted the old vector store')
            else:
                print(f'Provider `{self.db_provider} not supported.')

        # Create our Chroma vectorstore
        if db_provider == ProviderEnum.CHROMA:
            self.vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=self.db_name, collection_name=self.COLLECTION_NAME)
            print(f'The embedding file stored at {db_name}/chroma.sqlite3')
            print(f"Vectorstore created with {self.vectorstore._collection.count()} documents")
        elif db_provider == ProviderEnum.FAISS:
            self.vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            self.vectorstore.save_local(folder_path=db_name, index_name=self.EMBEDDING_INDEX_NAME)
            print(f'The embedding files stored at {db_name}/{self.EMBEDDING_INDEX_NAME}.faiss, {db_name}/{self.EMBEDDING_INDEX_NAME}.pkl')
            total_vectors = self.vectorstore.index.ntotal
            dimensions = self.vectorstore.index.d
            print(f"There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")
        else:
            raise Exception(f'Provider `{db_provider}` not supported.')

    def get_vector(self):
        if self.db_provider == ProviderEnum.CHROMA:
            # Get one vector and find how many dimensions it has
            collection = self.vectorstore._collection
            sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
            dimensions = len(sample_embedding)
            print(f"The vectors have {dimensions:,} dimensions")
            return sample_embedding
        elif self.db_provider == ProviderEnum.FAISS:
            sample_embedding = self.vectorstore.index.reconstruct(0)
            return sample_embedding
        raise Exception(f'Provider `{self.db_provider}` not supported.')

    def extract(self):
        if self.db_provider == ProviderEnum.CHROMA:
            return self.extract_chroma()
        elif self.db_provider == ProviderEnum.FAISS:
            return self.extract_faiss()
        raise Exception(f'Provider `{self.db_provider}` not supported.')

    def extract_chroma(self):
        # Prework
        result = self.vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
        colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]
        return vectors, documents, doc_types, colors

    def extract_faiss(self):
        # Prework
        vectors = []
        documents = []
        doc_types = []
        colors = []
        color_map = {'products':'blue', 'employees':'green', 'contracts':'red', 'company':'orange'}

        total_vectors = self.vectorstore.index.ntotal

        for i in range(total_vectors):
            vectors.append(self.vectorstore.index.reconstruct(i))
            doc_id = self.vectorstore.index_to_docstore_id[i]
            document = self.vectorstore.docstore.search(doc_id)
            documents.append(document.page_content)
            doc_type = document.metadata['doc_type']
            doc_types.append(doc_type)
            colors.append(color_map[doc_type])

        vectors = np.array(vectors)

        return vectors, documents, doc_types, colors

    def visualize2D(self, vectors: np.ndarray, colors: list=[], doc_types: set={}, documents=None):
        # We humans find it easier to visalize things in 2D!
        # Reduce the dimensionality of the vectors to 2D using t-SNE
        # (t-distributed stochastic neighbor embedding)

        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        # Create the 2D scatter plot
        fig = go.Figure(data=[go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title=f'2D {self.db_provider} Vector Store Visualization',
            scene=dict(xaxis_title='x', yaxis_title='y'),
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        fig.show()

    def visualize3D(self, vectors: np.ndarray, colors: list=[], doc_types: set={}, documents=None):
        # Let's try 3D!
        tsne = TSNE(n_components=3, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title=f'3D {self.db_provider} Vector Store Visualization',
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        fig.show()

    def chat(self, message: str, history=None):
        result = self.conversation_chain.invoke({"question": message})
        return result["answer"]


if __name__ == '__main__':
    # view = gr.ChatInterface(chat, type="messages").launch()

    # rag = RAG(db_provider=ProviderEnum.CHROMA, embedding_provider=ProviderEnum.HUGGING_FACE)
    # rag = RAG(db_provider=ProviderEnum.CHROMA, embedding_provider=ProviderEnum.OPENAI)
    # rag = RAG(db_provider=ProviderEnum.FAISS, embedding_provider=ProviderEnum.HUGGING_FACE)
    rag = RAG(db_provider=ProviderEnum.FAISS, embedding_provider=ProviderEnum.OPENAI)
    view = gr.ChatInterface(rag.chat, type="messages").launch(inbrowser=True)

