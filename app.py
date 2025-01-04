import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai


genai.configure(api_key='AIzaSyDa_MOq04fLAGFHQcxud5mAXQ_0BdKS3uA')


def process_text_with_gemini(text):
    
    chunk_size = 1000
    chunk_overlap = 200

    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])

    
    embeddings = []
    for chunk in chunks:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk
        )
        embeddings.append(result["embedding"])

    return chunks, embeddings


def main():
    st.title("Query PDF Document")

    
    pdf = st.file_uploader("Upload your PDF File", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

    
        chunks, embeddings = process_text_with_gemini(text)
        st.write("PDF processed successfully!")

        
        query = st.text_input("Ask a question about the PDF...")

        if query:
            
            from numpy import dot
            from numpy.linalg import norm

            def cosine_similarity(vec1, vec2):
                return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

            
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=query
            )["embedding"]

            
            similarities = [cosine_similarity(query_embedding, embedding) for embedding in embeddings]
            best_chunk_index = similarities.index(max(similarities))
            best_chunk = chunks[best_chunk_index]

            
            st.write("Most relevant text from the PDF:")
            st.write(best_chunk)

if __name__ == "__main__":
    main()

