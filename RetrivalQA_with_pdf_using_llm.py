from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import openai


from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import os

load_dotenv()
def read_pdf(pdf_file_path):
    """Function to read only PDF files """
    try:
        if not pdf_file_path.endswith(".pdf"):
            raise ValueError("File type not supported. Please use PDF files.")

        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        print("PDF read successfully .")
        return documents

    except ValueError:
        print(f"Error: PDF file not found at the specified path: {pdf_file_path}")
        return None

    except Exception as e:
        print(f"Error while reading PDF: {e}")
        return None

def model(pdf_file_path, question):
    """Function to answer the questions from  PDF only"""

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        document = read_pdf(pdf_file_path)
        if document is None:
            return ("Error while reading PDF")
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(document)

        # Select which embeddings we want to use
        embeddings = OpenAIEmbeddings()

        # Create the vector store to use as the index
        db = Chroma.from_documents(texts, embeddings)

        # Expose this index in a retriever interface
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Create a chain to answer questions
        llm = OpenAI(api_key=openai_api_key)

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="map_reduce", retriever=retriever
        )
        result = qa({"query": f'" {question} "'})
        return result

    except Exception as e:
        print(f"Error in model function: {e}")
        return None


pdf_file_path = r"C:\python\attention_all_you_need.pdf"
question = "how many identical layers are in Encoder "
print(model(pdf_file_path, question))