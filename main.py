import os
import streamlit as st
import pickle
import time
import langchain
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document 
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from langchain.vectorstores import FAISS


from dotenv import load_dotenv
# Load the environment variables from the .env file
load_dotenv()

st.title("News Research Tool")

st.sidebar.title("News Research URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process links")
file_path = "vector_index.pkl"

main_placeholder = st.empty()


repo_id = "google/flan-t5-small"
task = "text2text-generation"

# Create HuggingFacePipeline instance
llm = HuggingFaceHub(
    repo_id=repo_id,
    task=task,
    model_kwargs={"temperature": 0.9, "max_length": 100}
)



if process_url_clicked:
    if urls:  # Ensure there are valid URLs
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started ✅✅✅")
        data = loader.load()

        # Check if data is loaded correctly
        if not data:
            main_placeholder.text("No data loaded from the provided URLs.")
        else:
            # Add source metadata to each document
            docs = [Document(page_content=doc.page_content, metadata={"source": url}) for url, doc in zip(urls, data)]

            # Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ' '],
                chunk_size=1000,
            )
            main_placeholder.text("Text Splitter...Started ✅✅✅")
            split_docs = text_splitter.split_documents(docs)

            # Create embeddings
            embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            main_placeholder.text("Creating FAISS index... ✅")
            vectorstore = FAISS.from_documents(split_docs, embeddings)

            # Store vector index in local storage
            with open(file_path, 'wb') as f:
                pickle.dump(vectorstore, f)
            main_placeholder.text("Vector index created and saved ✅")

    else:
        main_placeholder.text("Please provide valid URLs.")

query = main_placeholder.text_input("Question: ")
if query:
    # Check if input exceeds token limits
    input_tokens = len(query.split())

    # Adjust max_new_tokens as necessary
    if input_tokens + 100 > 1024:  
        st.warning("Input too long. Please shorten your question.")
    else:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)



    


