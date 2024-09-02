#installing necessary librairies
from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain.embeddings import HuggingFaceEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from langchain_core.prompts.chat import ChatPromptTemplate
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from langchain.llms import HuggingFaceHub # Import OpenAI LLM
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
from deep_translator import GoogleTranslator
from langdetect import detect

# functions definitions

def load_documents(path):
  """
  Load PDF documents from the specified directory using PyPDFDirectoryLoader.
  Returns:
  List of Document objects: Loaded PDF documents represented as Langchain
                                                          Document objects.
  """
  # Initialize PDF loader with specified directory
  document_loader = PyPDFDirectoryLoader(path) 
  # Load PDF documents and return them as a list of Document objects
  return document_loader.load() 

def split_text(documents: list[Document]):
  """
  Split the text content of the given list of Document objects into smaller chunks.
  Args:
    documents (list[Document]): List of Document objects containing text content to split.
  Returns:
    list[Document]: List of Document objects representing the split text chunks.
  """
  # Initialize text splitter with specified parameters
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Size of each chunk in characters
    chunk_overlap=100, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
  )

  # Split documents into smaller chunks using text splitter
  chunks = text_splitter.split_documents(documents)
  return chunks # Return the list of split text chunks




 #saving documents to out chroma database
 # Path to the directory to save Chroma database

def save_to_chroma(chunks: list[Document],chroma_path):
  """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

  # Clear out the existing database directory if it exists
  if not os.path.exists(chroma_path):
    db = Chroma.from_documents(
    chunks,
    HuggingFaceEmbeddings(),
    persist_directory=chroma_path
  )

  # Persist the database to disk
    db.persist()  

def query_rag(query_text,chroma_path):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_text (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
    #Define Template
  PROMPT_TEMPLATE = """You are a fertilizer expert at OCP Group. These Human will ask you a questions about OCP Fertilizer. 
    Use following piece of context to answer the question. 
    If you don't know the answer, just say that you are an assistance tool,
    but it looks like I don’t have the specific information you’re looking for. 
    I recommend reaching out to the support team for more detailed assistance.
    They will be happy to help you further!. 
    make the sentence structured check if the generated text ends with a complete sentence.
    If not,continue generating text until a complete sentence is formed.

    Context: {context}
    Question: {question}
    Answer: 

    """

  # YOU MUST - Use same embedding function as before
  embedding_function = HuggingFaceEmbeddings()

  # Prepare the database
  db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Retrieving the context from the DB using similarity search
  results = db.similarity_search_with_relevance_scores(query_text, k=3)
  # Combine context from matching documents
  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

  # Create prompt template using context and query text
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)

  #intilize chatmodel
  repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
  model = HuggingFaceHub(
  repo_id=repo_id, 
  model_kwargs={"temperature": 0.8, "top_k": 50,
                "max_length": 200000, 
                "min_length": 50,}, 
     huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
  )

  # Generate response text based on the prompt
  response_text = model.predict(prompt)
  return response_text

def truncate_from_keyword(response, keyword):
    # Find the position of the keyword in the text
    keyword_pos = response.find(keyword)
    
    # If the keyword is found, truncate the text from the keyword to the end
    if keyword_pos != -1:
        response =response[keyword_pos + len(keyword):]
        return response
    else:
        return response.trim()


def translate_text(text, src_lang='en', dest_lang='fr'):
    translated = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    return translated

def detect_language(user_question,llm_response):
    language = detect(user_question)
    if language == 'fr':
       final_response = translate_text(llm_response,'en','fr')
    else:
       final_response = llm_response
    return final_response

def generate_final_response(response):
    #the model complete the sentence
    last_period_index= response.rfind('.')
    if last_period_index == -1:
        # No period found, return the truncated text as is
        return response
    return response[:last_period_index + 1]

if __name__ == "__main__":
  data_path = './Fertilizers'
  chroma_path = 'chroma'
  query_text = input("your question here")
  documents = load_documents(data_path)
  chunks = split_text(documents)
  save_to_chroma(chunks, chroma_path)
  #Env Variables
  load_dotenv()
  response = query_rag(query_text,chroma_path)
  truncate_response = truncate_from_keyword(response,'Answer:')
  translate_response = detect_language(query_text,truncate_response)
  final_response = generate_final_response(translate_response)
  print(final_response)
  
  