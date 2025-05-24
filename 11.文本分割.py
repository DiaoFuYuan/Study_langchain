from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

data_doc = UnstructuredLoader(r"D:\ai\study\file\处罚决定书（四川华筑联投置业有限公司）.docx")
doc = data_doc.load()
print("文档加载")
print(doc)

doc_splitter = CharacterTextSplitter(
	separator="\n\n",
	chunk_size=100,
	chunk_overlap=10,
	length_function=len,
	is_separator_regex=False,
)

for document in doc:
	texts = doc_splitter.split_text(document.page_content)
	print("文本分割")
	print(texts)


# texts = doc_splitter.create_documents()

