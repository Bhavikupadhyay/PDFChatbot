import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import getpass

def st_app():
    st.title('PDF Chatbot')

    uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf'], accept_multiple_files=False)

    if uploaded_file is not None:
        st.write('Uploaded File:', uploaded_file.name)
        
        temp_folder = '/tmp'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        tmp_path = os.path.join(temp_folder, uploaded_file.name)
        with open(tmp_path, 'wb') as tmp_file:
            tmp_file.write(uploaded_file.read())
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        os.remove(tmp_path)

        user_question = st.text_input('Ask a question:', max_chars=200)
        response = None

        splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0.1)
        text = splitter.split_documents(documents=docs)

        embeddings = HuggingFaceEmbeddings()
        db = Chroma.from_documents(text, embeddings)
        retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 3})

        repo_id = 'google/flan-t5-small'
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs = {'temperature': 0.3, 'max_length': 200}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='map_reduce',
            retriever=retriever,
            return_source_documents=False,
        )

        if user_question:
            response = qa_chain({'query': user_question})['result']

        st.write('Chatbot Response:')
        if response is not None:
            st.markdown(response)


if __name__ == '__main__':
    TOKEN = getpass.getpass('Please enter your HuggingFace API Token:')

    os.environ['HUGGINGFACEHUB_API_TOKEN'] = TOKEN
    st_app()