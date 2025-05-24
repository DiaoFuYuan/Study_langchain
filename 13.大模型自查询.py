# 自查询：使用大模型根据用户的问题，进行扩展在从数据库中检索相关数据。

from langchain.retrievers import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader,PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import uuid
import os
import nltk
nltk.data.path.append(r"D:\ai\study\nltk_data")
import lark
