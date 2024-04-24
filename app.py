# import libraries
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
import os

# get the creds
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "secrets/glossy-attic-415618-93704b0714e2.json"

TEMP_FILE_PATH = "temp.txt"

llm = VertexAI(
    model_name="gemini-pro",
    project="glossy-attic-415618",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,)
    
embeddings = VertexAIEmbeddings()


def get_text(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Create a BeautifulSoup object with the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the specific element or elements containing the text you want to scrape
    # Here, we'll find all <p> tags and extract their text
    paragraphs = soup.find_all("p")

    # Loop through the paragraphs and print their text
    with open(TEMP_FILE_PATH, "w", encoding='utf-8') as file:
        # Loop through the paragraphs and write their text to the file
        for paragraph in paragraphs:
            file.write(paragraph.get_text() + "\n")

@st.cache_resource
def create_langchain_index(input_text):
    print("--indexing---")
    get_text(input_text)
    loader = TextLoader(TEMP_FILE_PATH, encoding='utf-8')
    # data = loader.load()

    index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch,embedding=embeddings).from_loaders([loader])
    return index

@st.cache_data
def get_response(input_text,query):
    print(f"--querying---{query}")
    response = index.query(query,llm=llm)
    return response


st.set_page_config(page_title="AI News Summary",page_icon=":newspaper:")

st.title('News Article Summary and Q&A')

st.write("By: Alex")
st.write("A Work-In-Progress project for Google Generative AI Hackathon https://googleai.devpost.com/")

st.write("")

input_text=st.text_input("Provide the link to the webpage...")

summary_response = ""
tweet_response = ""
ln_response = ""
# if st.button("Load"):
if input_text:
    index = create_langchain_index(input_text)
    summary_query ="Write a 100 words summary of the document"
    summary_response = get_response(input_text,summary_query)

    objective_query ="Decide whether the author of the document is being objective or siding with one side. Provide an example."
    objective_response =  get_response(input_text,objective_query)

    ln_query ="Write a linkedin post for the document, and add an emoji for each main"
    ln_response = get_response(input_text,ln_query)


    with st.expander('Page Summary'): 
        st.info(summary_response)

    with st.expander('Objectivity'): 
        st.info(objective_response)

    with st.expander('LinkedIn Post'): 
        st.info(ln_response)


st.session_state.input_text = ''    
question=st.text_input("Ask a question from the link you shared...")
if st.button("Ask"):
        if question:
            index = create_langchain_index(input_text)
            response = get_response(input_text,question)
            st.write(response)
        else:
            st.warning("Please enter a question.")
    