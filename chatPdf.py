import streamlit as st
import PyPDF2
import logging
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
from openai.error import RateLimitError

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

OPENAI_API_KEY = "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"


def read_pdf(file):

    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = pdf_reader.pages
    logging.info(len(num_pages))

    langchain_pages = []
    for page_num in range(len(num_pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        lines = text.split('\n')
        
        for line in lines:
            doc = Document(page_content=line, metadata={'page': page_num})
            langchain_pages.append(doc)

    logging.info(f"Parsed {len(langchain_pages)} Langchain schema Documents")
    return langchain_pages

def get_embeddings():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,max_retries=1)
        return embeddings
    except RateLimitError as r:
        logging.info('Assigned rate limit exceeded for the given OPENAI API key.')
        st.warning("You have hit your assigned rate limit for your OPENAI account api key, please change it or renew the subscription.")


def main():

    st.title("Automating PDF Interaction with LangChain and ChatGPT")
    # st.write("Upload a PDF file")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    pages = []
    if uploaded_file is not None:
        logging.info('File uploaded successfully.')
        pages = read_pdf(uploaded_file)
    
    if pages:
        logging.info('Langchain pages list prepared.')
        
        openai_embeddings = get_embeddings()
        logging.info('OPENAI embedding fetched successfully.')

        vectordb = Chroma.from_documents(pages, embedding=openai_embeddings,
                                 persist_directory=".")
        logging.info('Vectorization done.')

        pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
                                    vectordb, return_source_documents=True)
        
    query = st.text_input("Enter your Query")
    if query:
        result = pdf_qa({"question": query, "chat_history": ""})
        st.write("ANSWER:\n")
        st.write(result['answer'])


if __name__ == "__main__":
    main()
