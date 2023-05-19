
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback




def main():
    load_dotenv()
    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf 💻')

    #upload pdf
    pdf=st.file_uploader('Upload your pdf', type='pdf')

    #extract text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #split text into chunks
        splitter=CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=splitter.split_text(text)
        
        #create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base=FAISS.from_texts(chunks, embeddings)

        #show input
        input=st.text_input('Ask your question about your pdf')
        if input:
            #get answer
            docs=knowledge_base.similarity_search(input)

            llm=OpenAI()
            chain=load_qa_chain(llm,chain_type="stuff")
            
            answer=chain.run(input_documents=docs,question=input)
            st.write(answer)


if __name__ == "__main__":
    main()