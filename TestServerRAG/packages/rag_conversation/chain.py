import os
from operator import itemgetter
from typing import List, Tuple

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
openai_api_key=os.getenv("OPENAI_API_KEY")
pinecone_api_key=os.getenv("PINECONE_API_KEY")

if not langchain_api_key or not openai_api_key or not pinecone_api_key:
    raise EnvironmentError("As chaves 'LANGCHAIN_API_KEY', 'OPENAI_API_KEY' e 'PINECONE_API_KEY' precisam ser definidas como variáveis de ambiente.")

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "pes")

llm = ChatOpenAI(model="gpt-4")


### Ingest code - you may need to run this the first time
# # Load
#  from langchain_community.document_loaders import PyPDFLoader
#  loader = PyPDFLoader("pleceholder")
#  data = loader.load()

# # Split
#  from langchain_text_splitters import RecursiveCharacterTextSplitter
#  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#  all_splits = text_splitter.split_documents(data)

# # Add to vectorDB
#  vectorstore = PineconeVectorStore.from_documents(
#      documents=all_splits, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
#  )

vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, OpenAIEmbeddings()
)

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=5)  # select top result

# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Você é um assistente de respostas utíl, especializado na metodologia PES (Planejamento Estratégico Situacional) que ajuda a analisar casos. 
Antes de você entregar as sua respostas, faça um passo a passo da sua linha de raciocínio para só então criar uma resposta estruturada. 
Você prioriza o uso de linguagem técnica da área sobre linguajar comum. 
Se estiver incerto sobre como responder algum comando, especifique qual parte da pergunta gerou incerteza e recuse-se a responder com a frase 'Infelizmente não tenho dados o bastante para responder essa dúvida'. 
Faça uso do seguinte contexto conforme for relevante: {context}"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = _inputs | ANSWER_PROMPT | llm | StrOutputParser()
