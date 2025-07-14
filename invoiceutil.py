import os
import tempfile
from io import BytesIO
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import re

load_dotenv()

def create_docs(user_pdf_list):
    """Extracts structured invoice data using Google Gemini (Gemini Pro)."""
    all_extracted_data = []  # List to store results from multiple PDFs

    for uploaded_file in user_pdf_list:
        print("Processing:", uploaded_file.name)

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

            # Load PDF content using PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()

            # Embeddings using Gemini
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Create FAISS vector store
            vector = FAISS.from_documents(pages, embeddings)

            # Prompt template for Gemini
            template = """
             Extract the following values: invoice no., Description, Quantity, date, Unit price, Amount,
             Total, email, phone number, and address from the following invoice content:
             {context}

             Return ONLY the JSON output with the extracted fields. Do NOT include any markdown, code block markers (e.g., ```json
             Format example:
             {{"Invoice no.": "XXXX", "Description": "ABC", "Quantity": "2", "Date": "01/01/2024", "Unit Price": "100.00", "Amount": "200.00", "Total": "200.00", "Email": "abc@xyz.com", "Phone Number": "1234567890", "Address": "123 Street, City, Country"}}

             Remove any currency symbols from numeric fields.
            """
            
            prompt = PromptTemplate.from_template(template)

            # Gemini LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

            # Retrieval chain setup
            retriever = vector.as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Run the chain
            response = retrieval_chain.invoke({"input": "Extract invoice details"})
            answer_content = response['answer']

            # Clean the response to remove any markdown or extra text
            answer_content = re.sub(r'```json\s*|\s*```', '', answer_content).strip()
            print("Extracted Data:", answer_content)

            all_extracted_data.append({uploaded_file.name: answer_content})

            # Clean up temporary file
            os.unlink(tmp_file_path)

    print("************* ALL FILES PROCESSED *************")
    return all_extracted_data