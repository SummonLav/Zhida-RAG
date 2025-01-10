# %% [markdown]
# # Build a Retrieval Augmented Generation (RAG) Model for Zhida Technology
# 
# Document from 港交所上市文件-挚达科技-业务【https://www1.hkexnews.hk/app/sehk/2024/106935/2024112801952_c.html】
# 
# Reference Links to Learn LangChain:
# 1. DeepLearning.ai Course: https://learn.deeplearning.ai/courses/langchain-chat-with-your-data
# 2. LangChain Github: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials

# %% [markdown]
# ## Retrieval augmented generation
#  
# In retrieval augmented generation (RAG), an LLM retrieves contextual documents from an external dataset as part of its execution. 
# 
# This is useful if we want to ask question about specific documents (e.g., our PDFs, a set of videos, etc).

# %% [markdown]
# ### Document Loading
# 
# Reference: DeepLearning.ai -> https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/2/document-loading
# 
# Document Reference: 01_document_loading.ipynb

# %%
import os
import getpass
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
openai.api_key  = os.environ['OPENAI_API_KEY']

# %% [markdown]
# Load PDF (*PyPDFLoader* by *langchain.document_loaders* does NOT support PDFs from 港交所)
# 
# Way 1: pdfplumber
# 

# %%
#%pip install pypdf

# %%
from langchain.document_loaders import PyPDFLoader
import pdfplumber

loader = pdfplumber.open("/Users/lavendashan/Documents/AIML/LangChain-RAG/Files/挚达-业务.pdf")
pages = loader.pages

# ！读取为乱码 ！#
#loader = PyPDFLoader("/Users/lavendashan/Documents/AIML/LangChain-RAG/Files/挚达-业务.pdf")
#pages = loader.load()

### TEST ###
#import pdfplumber
#with pdfplumber.open("/Users/lavendashan/Documents/AIML/LangChain-RAG/Files/挚达-业务.pdf") as pdf:
 #   for page in pdf.pages:
  #      text = page.extract_text()
   #     print(text)

# %%
len(pages)

# %%
### 第一页 ###
page = pages[0]

print(page.extract_text())

# %% [markdown]
# Way 2: LangChain PDFPlumberLoader
# 
# Reference: https://python.langchain.com/docs/integrations/document_loaders/pdfplumber/

# %%
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("/Users/lavendashan/Documents/AIML/LangChain-RAG/Files/挚达-业务.pdf")
docs = loader.load()

# %%
print(docs[0])

# %%
### 图片分析不了，只能读取里面的文本 ###
print(docs[3])

# %%
### 检查表格读取，应该可以 ###
print(docs[19])

# %%
# 提取所有文档的文本内容
all_text = " ".join([doc.page_content for doc in docs])
print(all_text[:500])  # 打印前500个字符作为示例

# %%
# 去掉空格，保留换行符
all_text = all_text.replace(" ", "")

print(all_text[:500])

# %% [markdown]
# ### Embedding and VectorStore

# %% [markdown]
# Use Way 2: *LangChain PDFPlumberLoader* to continue

# %% [markdown]
# 通常，**Embedding** 生成向量，**VectorStore** 存储向量并高效检索相关内容。
# 
# **Embedding** 和**Vectorize** 区别：<br>
# **Embedding** 更注重语义，是一种深层的、上下文感知的向量化方法，适合复杂任务（如语义搜索、推荐系统）。<br>
# **Vectorize** 是更通用的向量化技术，可以是简单统计、特征提取，或者深度学习生成的向量。

# %% [markdown]
# 又是报错 -- LangChain官方的text_splitter不能分割中文😅，那就试试jieba！

# %% [markdown]
# **RecursiveCharacterTextSplitter** vs. **Jieba** 区别 <br>
# **RecursiveCharacterTextSplitter**:
# - 按段落、句子或固定长度的块进行分割。
# - 保持上下文连续性，通常分割结果是较大的文本块（句子块或字符块），主要用于生成语言模型的输入片段。
# 
# **Jieba**:
# - 按单词粒度进行分词，将句子切割成最小的词汇单位。
# - 主要用于语义分析、信息检索等需要精准语义单位的场景。

# %%
import jieba

seg_list = jieba.cut(all_text, cut_all=False)
#print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

# %% [markdown]
# 感觉分词结果太细了，试试类LangChain RecursiveCharacterTextSplitter的按固定长度分块看看模型上下文理解的效果👀？<br>
# （chunk_size 和 chunk_overlap 与 LangChain 参考文档相同）
# 
# **可能问题**：
# 分词结果太细会导致向量数据库存储大量碎片，影响检索效果和生成的上下文质量。<br>
# **解决方案**：
# 将分词后的结果重新组织为更大的块（chunk），例如以一定的字数或句子数为单位进行合并。
# 
# 

# %%
chunk_size = 1500  # 每块最多包含 1500 个字符
chunk_overlap = 150

# 实现分块并加入重叠
chunks = [
    all_text[i:i + chunk_size]
    for i in range(0, len(all_text), chunk_size - chunk_overlap)
    if i < len(all_text)  # 确保块起始索引在文本范围内
]
print("按固定长度分块（含重叠）:", chunks)

# 对每个块进行分词
segmented_chunks = [" ".join(jieba.cut(chunk)) for chunk in chunks]
print("分词后（含重叠）:", segmented_chunks)

# %%
len(segmented_chunks[0])   # 检查 chunk_size

# %% [markdown]
# 将 segmented_chunks 转为 LangChain 文档对象，继续向量存储和检索的步骤

# %%
from langchain.schema import Document

# 将分词后的文本块转换为 LangChain 文档对象
splits = [Document(page_content=chunk) for chunk in segmented_chunks]
print("LangChain 文档对象:", splits[:2])

# %%
print("LangChain 文档对象:", splits[-1])

# %%
print(f"创建了 {len(splits)} 个 LangChain 文档。")

# %% [markdown]
# 将这些合并后的文档块存储到向量数据库中，方便后续检索

# %%
#!rm -rf /Users/lavendashan/Documents/AIML/LangChain-RAG/docs  # remove old database files if any

# %%
#!mkdir -p /Users/lavendashan/Documents/AIML/LangChain-RAG/docs/chroma

# %%
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

persist_directory = "/Users/lavendashan/Documents/AIML/LangChain-RAG/docs/chroma"
embedding = OpenAIEmbeddings()

# %%
# 存储到向量数据库
vectordb = Chroma.from_documents(
    documents=splits, 
    embedding=embedding, 
    persist_directory=persist_directory
)

# 持久化存储
vectordb.persist()
print("向量数据库已存储成功！")

# %%
print(vectordb._collection.count())

# %% [markdown]
# ### RAG-enabled ChatBot Interface

# %%
# 加载存储的向量数据库
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 测试检索
query = "挚达业务的主要内容是什么？"
results = retriever.get_relevant_documents(query)
for doc in results:
    print(doc.page_content)

# %%
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 使用 ChatOpenAI 代替 OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=1)

# 创建 RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    chain_type="refine"
)

# %%
# 测试问答
query = "请问这个文档的主要内容是什么？"
answer = qa_chain.run(query)
print(f"回答: {answer}")

# %%
import gradio as gr

def chatbot(user_input):
    return qa_chain.run(user_input)

# Gradio Web 聊天机器人
iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="挚达聊天机器人"
)

iface.launch()

# %% [markdown]
# ### 进一步调试

# %% [markdown]
# 1. 无法检索到机器人相关产品 -- 【解决方案】GPT3.5更新到4o

# %%
# 测试检索
query = "挚达科技与机器人相关的业务有哪些？"
results = retriever.get_relevant_documents(query)
for doc in results:
    print(doc.page_content)

# %%
# 测试问答
query = "挚达有哪些机器人产品？"
answer = qa_chain.run(query)
print(f"回答: {answer}")

# %% [markdown]
# 2. 非multimodal -- 无法读图

# %%
# 测试检索
query = "挚达的APP包含什么内容？"
results = retriever.get_relevant_documents(query)
for doc in results:
    print(doc.page_content)

# %%
# 测试问答
query = "挚达的APP包含什么内容？"
answer = qa_chain.run(query)
print(f"回答: {answer}")

# %% [markdown]
# 3. Hallucination (wrong answer) -- 【解决方案】在 qa_chain 加入 chain_type="refine" 的合并答案方式，解决了乱回答的问题但是回答时间变长 && 不能自动调用 ChatGPT 的 RAG (Possible Reason: By default, most of LangChain’s built-in RetrievalQA prompts encourage the LLM to say “I don’t know” if the retrieval doesn’t yield relevant context.)

# %%
# 测试问答
query = "挚达科技成立于哪一年？"
results = retriever.get_relevant_documents(query)
for doc in results:
    print(doc.page_content)

# %%
# 测试问答
query = "挚达科技成立于哪一年？"
answer = qa_chain.run(query)
print(f"回答: {answer}")

# %%


