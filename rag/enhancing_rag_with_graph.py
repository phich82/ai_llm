from datetime import datetime
import os

from typing import List
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever
from yfiles_jupyter_graphs import GraphWidget

from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.runnables.utils import Output

from pydantic import BaseModel, Field
from neo4j import GraphDatabase
# from neo4j import Driver

# Load all environment variables from .env file
load_dotenv()


class Entities(BaseModel):
    """Identifying information about entities."""
    # list: python 3.9, typings.List: python 3.8
    # names: list[str] = Field(
    #     ...,
    #     description="All the person, organization, or business entities that appear in the text",
    # )
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

class DB:

    NEO4J_URI = 'bolt://localhost:7687'
    NEO4J_USERNAME = 'neo4j'
    NEO4J_PASSWORD = 'neo4j'

    # Create database connection (Neo4J Database)
    graph: Neo4jGraph = None
    llm = None
    vector_retriever: VectorStoreRetriever = None

    def __init__(self, url: str='', username: str='', password: str=''):
        url = url if url is not None and url != '' else os.environ["NEO4J_URI"]
        username = username if username is not None and url != '' else os.environ["NEO4J_USERNAME"]
        password = password if password is not None and password != '' else os.environ["NEO4J_PASSWORD"]

        if self.graph is None:
            self.graph = Neo4jGraph(
                url=url,
                username=username,
                password=password
            )

    def load_documents(self, document_path: str='', chunk_size: int=250, chunk_overlap: int=24, llm_type: str='ollama'):
        # Load text (documents)
        loader = TextLoader(file_path=document_path)
        docs = loader.load()
        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents=docs)
        print(documents)

        current_date = datetime.now()
        start_time: int = current_date.microsecond
        print(f'Start date ======> {current_date} ')
        # Load model (Ollama: ollama MUST be installed previous and model (llama3.1:8b) MUST be pulled)
        # Ref: https://ollama.com/library/llama3.1:8b => ollama run llama3.1:8b
        # Ref: https://ollama.com/ => download OllamaSetup.exe
        llm_type = llm_type if llm_type is not None and llm_type != '' else os.getenv('LLM_TYPE', 'ollama')
        print('llm_type => '+ llm_type)
        # llm = ChatOllama(model='llama3.1', temperature=0) if llm_type == 'ollama' else ChatOpenAI(model='gpt-4.1', temperature=0)
        if llm_type == 'ollama':
            self.llm = ChatOllama(model='llama3.1:8b', temperature=0)
            # llm = OllamaFunctions(model="llama3.1:8b", temperature=0, format="json")
        else:
            self.llm = ChatOpenAI(model='gpt-4.1', temperature=0)
        llm_transformer = LLMGraphTransformer(llm=self.llm)

        # Convert to graph documents
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        print(graph_documents[0])

        end_time = datetime.now()
        print(f'End date ======> {end_time} ')
        print(f'Processing date (convert_to_graph_documents) ======> {(end_time.microsecond - start_time) / 1000} ')

        # Save to database
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        print(f'Processing date (add_graph_documents) ======> {datetime.now()} ')

    # Show large knowledge graph
    def showGraph(self):
        driver = GraphDatabase.driver(
            uri = self.NEO4J_URI,
            auth = (self.NEO4J_USERNAME, self.NEO4J_PASSWORD)
        )
        session = driver.session()
        widget: GraphWidget = GraphWidget(graph=session.run('MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t').graph())
        widget.node_label_mapping = 'id'
        return widget

    def create_index(self):
        # Create vector store / search vector
        # If not exist => ollama pull mxbai-embed-large
        # embeddings = OllamaEmbeddings(
        #     model="mxbai-embed-large",
        # )
        embeddings = OpenAIEmbeddings()
        vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        self.vector_retriever = vector_index.as_retriever()
        print(f'Processing date (embedding) ======> {datetime.now()} ')

        # #############################################################
        driver = GraphDatabase.driver(
            uri = self.NEO4J_URI,
            auth = (self.NEO4J_USERNAME, self.NEO4J_PASSWORD)
        )

        # Create index
        with driver.session() as session:
            session.execute_write(self.create_fulltext_index)
            print("Fulltext index created successfully.")

        # Close the driver connection
        driver.close()
        # ###########################################################
        print(f'Processing date (create_index) ======> {datetime.now()} ')

    def create_fulltext_index(self, tx):
        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_id` 
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        tx.run(query)


class ChatAI:

    db: DB

    def __init__(self, db: DB=None):
        self.db = db if db is None else DB()

    def generate_full_text_query(self, input: str) -> str:
        words = [el for el in remove_lucene_chars(input).split() if el]
        if not words:
            return ""
        full_text_query = " AND ".join([f"{word}~2" for word in words])
        print(f"Generated Query: {full_text_query}")
        return full_text_query.strip()

    # Fulltext index query
    def graph_retriever(self, question: str, chain: ChatPromptTemplate) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = chain.invoke(question)
        for entity in entities.names:
            response = self.db.graph.query(
                """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def full_retriever(self, question: str, chain: ChatPromptTemplate):
        graph_data = self.graph_retriever(question, chain=chain)
        vector_data = [el.page_content for el in self.db.vector_retriever.invoke(question)]
        final_data = f"""
        Graph data:{graph_data}
        vector data: {"#Document ". join(vector_data)}
        """
        return final_data

    def chat(self, question: str=""):
        print(f'[chat] start ======> {datetime.now()}')
        # 1. Using prompt messages
        prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )
        chain = prompt_template | self.db.llm.with_structured_output(Entities)
        # Output
        output: Output = chain.invoke({"question": question}) # ~ entity_chain.invoke(question)
        print(f'[chat] output ===> {output}')
        print(f'[chat] names ===> {output.names}')
        print("[chat] graph_retriever ===> " + self.graph_retriever(question=question, chain=chain))

        # 2. Using template message
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = (
            {
                "context": self.full_retriever(question=question, chain=chain), # self.full_retriever
                "question": RunnablePassthrough(),
            }
            | prompt_template
            | self.db.llm
            | StrOutputParser()
        )
        # Output
        output: Output = chain.invoke({'question': question}) # ~ chain.invoke(input=question)
        print(f'[chat] output: {output}')
        print(f'[chat] end ======> {datetime.now()} ')


if __name__ == '__main__':

    # NEO4J_URI = os.environ["NEO4J_URI"]
    # NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
    # NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
    # print(f'NEO4J_URI ==> {NEO4J_URI}')
    # print(f'NEO4J_USERNAME ==> {NEO4J_USERNAME}')
    # print(f'NEO4J_PASSWORD ==> {NEO4J_PASSWORD}')

    # # Create database connection (Neo4J Database)
    # graph = Neo4jGraph(
    #     # url=NEO4J_URI,
    #     # username=NEO4J_USERNAME,
    #     # password=NEO4J_PASSWORD
    # )

    # # Load text (documents)
    # loader = TextLoader(file_path="dummytext_shorten.txt")
    # docs = loader.load()

    # # Split text into smaller chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    # documents = text_splitter.split_documents(documents=docs)
    # print(documents)

    # current_date = datetime.now()
    # start_time: int = current_date.microsecond
    # print(f'Start date ======> {current_date} ')
    # # Load model (Ollama: ollama MUST be installed previous and model (llama3.1:8b) MUST be pulled)
    # # Ref: https://ollama.com/library/llama3.1:8b => ollama run llama3.1:8b
    # # Ref: https://ollama.com/ => download OllamaSetup.exe
    # llm_type = os.getenv('LLM_TYPE', 'ollama')
    # print('llm_type => '+ llm_type)
    # # llm = ChatOllama(model='llama3.1', temperature=0) if llm_type == 'ollama' else ChatOpenAI(model='gpt-4.1', temperature=0)
    # if llm_type == 'ollama':
    #     llm = ChatOllama(model='llama3.1:8b', temperature=0)
    #     # llm = OllamaFunctions(model="llama3.1:8b", temperature=0, format="json")
    # else:
    #     llm = ChatOpenAI(model='gpt-4.1', temperature=0)
    # llm_transformer = LLMGraphTransformer(llm=llm)

    # # Convert to graph documents
    # graph_documents = llm_transformer.convert_to_graph_documents(documents)
    # print(graph_documents[0])

    # end_time = datetime.now()
    # print(f'End date ======> {end_time} ')
    # print(f'Processing date (convert_to_graph_documents) ======> {(end_time.microsecond - start_time) / 1000} ')

    # #
    # graph.add_graph_documents(
    #     graph_documents,
    #     baseEntityLabel=True,
    #     include_source=True
    # )
    # print(f'Processing date (add_graph_documents) ======> {datetime.now()} ')

    # # Show large knowledge graph
    # def showGraph():
    #     driver = GraphDatabase.driver(
    #         uri = os.environ["NEO4J_URI"],
    #         auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    #     )
    #     session = driver.session()
    #     widget: GraphWidget = GraphWidget(graph=session.run('MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t').graph())
    #     widget.node_label_mapping = 'id'
    #     return widget

    # print(showGraph())

    # # Create vector store / search vector
    # # If not exist => ollama pull mxbai-embed-large
    # # embeddings = OllamaEmbeddings(
    # #     model="mxbai-embed-large",
    # # )
    # embeddings = OpenAIEmbeddings()
    # vector_index = Neo4jVector.from_existing_graph(
    #     embeddings,
    #     search_type="hybrid",
    #     node_label="Document",
    #     text_node_properties=["text"],
    #     embedding_node_property="embedding"
    # )
    # vector_retriever = vector_index.as_retriever()
    # print(f'Processing date (embedding) ======> {datetime.now()} ')

    # # #############################################################
    # driver = GraphDatabase.driver(
    #     uri = os.environ["NEO4J_URI"],
    #     auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    # )

    # def create_fulltext_index(tx):
    #     query = '''
    #     CREATE FULLTEXT INDEX `fulltext_entity_id` 
    #     FOR (n:__Entity__) 
    #     ON EACH [n.id];
    #     '''
    #     tx.run(query)

    # # # Function to execute the query
    # def create_index():
    #     with driver.session() as session:
    #         session.execute_write(create_fulltext_index)
    #         print("Fulltext index created successfully.")

    # # Call the function to create the index
    # try:
    #     create_index()
    # except:
    #     pass

    # # Close the driver connection
    # driver.close()
    # # ###########################################################
    # print(f'Processing date (create_index) ======> {datetime.now()} ')

    db = DB()
    # db.load_documents(document_path='dummytext_shorten.txt')
    question = "Who are Nonna Lucia and Giovanni Caruso?"
    # question = "Who is Nonna Lucia?"
    # question = "Who is Nonna Lucia? Did she teach anyone about restaurants or cooking?"
    chatAI = ChatAI(db=db)
    chatAI.chat(question=question)

    # template = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are extracting organization and person entities from the text.",
    #         ),
    #         (
    #             "human",
    #             "Use the given format to extract information from the following "
    #             "input: {question}",
    #         ),
    #     ]
    # )

    # entity_chain = template | db.llm.with_structured_output(Entities)
    # # print(f'entity_chain ===> {entity_chain.invoke("Who are Nonna Lucia and Giovanni Caruso?").names}')
    # print(f'entity_chain ===> {entity_chain.invoke({"question": "Who are Nonna Lucia and Giovanni Caruso?"}).names}')
    # print(f'Processing date (entity_chain) ======> {datetime.now()} ')

    # def generate_full_text_query(input: str) -> str:
    #     words = [el for el in remove_lucene_chars(input).split() if el]
    #     if not words:
    #         return ""
    #     full_text_query = " AND ".join([f"{word}~2" for word in words])
    #     print(f"Generated Query: {full_text_query}")
    #     return full_text_query.strip()

    # # Fulltext index query
    # def graph_retriever(question: str) -> str:
    #     """
    #     Collects the neighborhood of entities mentioned
    #     in the question
    #     """
    #     result = ""
    #     entities = entity_chain.invoke(question)
    #     for entity in entities.names:
    #         response = db.graph.query(
    #             """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
    #             YIELD node,score
    #             CALL {
    #             WITH node
    #             MATCH (node)-[r:!MENTIONS]->(neighbor)
    #             RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
    #             UNION ALL
    #             WITH node
    #             MATCH (node)<-[r:!MENTIONS]-(neighbor)
    #             RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
    #             }
    #             RETURN output LIMIT 50
    #             """,
    #             {"query": entity},
    #         )
    #         result += "\n".join([el['output'] for el in response])
    #     return result

    # print("graph_retriever ===> " + graph_retriever("Who is Nonna Lucia?"))


    # #####################
    # def full_retriever(question: str):
    #     graph_data = graph_retriever(question)
    #     vector_data = [el.page_content for el in db.vector_retriever.invoke(question)]
    #     final_data = f"""
    #     Graph data:{graph_data}
    #     vector data: {"#Document ". join(vector_data)}
    #     """
    #     return final_data

    # #####################
    # template = """Answer the question based only on the following context:
    # {context}

    # Question: {question}
    # Use natural language and be concise.
    # Answer:"""
    # template = ChatPromptTemplate.from_template(template)

    # chain = (
    #     {
    #         "context": full_retriever,
    #         "question": RunnablePassthrough(),
    #     }
    #     | template
    #     | db.llm
    #     | StrOutputParser()
    # )
    # print(f'Processing date (chain) ======> {datetime.now()} ')

    # # 
    # print(chain.invoke(input="Who is Nonna Lucia? Did she teach anyone about restaurants or cooking?"))
    # print(f'Processing date (chain:invoke) ======> {datetime.now()} ')

    # print("================> done")