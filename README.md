# A Demo of Retrieval Augmented Generation (RAG) with a Chat Interface 💬

This demo presents a Retrieval Augmented Generation (RAG) model that addresses questions about Zhida Tech’s documents, as posted on The Stock Exchange of Hong Kong Limited. Observations indicate that the OpenAI GPT4o model was not trained on this specific content, leading to difficulties in providing certain document-specific answers. 

Both the model’s strengths and its limitations are highlighted and discussed in the .ipynb file.

Comparison:
1. Question 1 appearing in Zhida Tech’s documents - RAG excels with its charmingly concise response
<div style="display: flex; justify-content: space-around;">
  <img width="450" alt="文档标准问题1-RAG" src="https://github.com/user-attachments/assets/29b96f69-58df-4647-8459-acdf354658d9" />
  <img width="450" alt="文档标准问题1-ChatGPT" src="https://github.com/user-attachments/assets/1e9b1c37-5cf3-4867-8776-3723dcc8e2ab" />
</div>

</br>

2. Question 2 appearing in Zhida Tech’s documents - RAG excels and the answer perfectly aligns with the document paragraphs
<div style="display: flex; justify-content: space-around;">
  <img width="450" alt="文档标准问题2-RAG" src="https://github.com/user-attachments/assets/8ccd3bf1-bc92-430c-b9e2-6162aff3f191" />
  <img width="450" alt="文档标准问题2-ChatGPT" src="https://github.com/user-attachments/assets/11d95e11-0a5b-4f4c-ae52-d7db6cfb8224" />
</div>

</br>

Limitations:
1. Hallucination
<div style="display: flex; justify-content: space-around;">
  <img width="450" alt="limitation2-无memory" src="https://github.com/user-attachments/assets/51881be7-ad68-42e4-99b0-144ac6c20bc8" />
  <img width="450" alt="limitation3-ChatGPT-正确" src="https://github.com/user-attachments/assets/decd87d7-2101-4196-9cde-f23c1205581f" />
</div>

