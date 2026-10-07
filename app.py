import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()
st.title("Document Based Q/A")

uploaded_file = st.file_uploader(
    "Upload Your Document as PDF",
    type=["pdf"]
)

if uploaded_file :
    with open(uploaded_file.name,"wb") as f: # creates new file in disk with same name
        f.write(uploaded_file.getbuffer()) # writes everything in new file which is copied from uploaded file

    loader = PyPDFLoader(uploaded_file.name) # initialize the pdf file path 

    documents = loader.load() # load the pdf in memory 

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(# assigning vetorstore as retriever
         search_kwargs = {"k":1}
    ) 

    question = st.text_input(
        "Ask your QUESTION related to the PDF file"
    )

    if question:
        relevant_docs = retriever.invoke(question)

        context = "\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature=0
        )

        prompt = f"""use only the context below.
        
        context :{context}
        question :{question}"""

        response = llm.invoke(prompt)

        st.write(response.content)
        st.subheader("Retrieved Chunks Related to Question is:")
        st.text(context)
