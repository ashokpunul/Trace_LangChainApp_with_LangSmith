#1) Data Ingestion (Document Loaders)
    #Data Ingestion means bringing data into our program so the computer can use it.

    #pip install langchain langchain-community

    #e.g:
text="""LangChain is a framework for building AI apps.
It helps connect language models with your own data.
RAG means Retrieval-Augmented Generation. """

with open("sample.txt", "w") as f:
    f.write(text)
print("sample.txt created successfully.")


#Loader:
    #A loader reads the file and turns it into Documents that LangChain understands.
    #pip install TextLoader
from langchain_community.document_loaders import TextLoader
loader = TextLoader("sample.txt")
#TextLoader	-->A .txt file
#PyPDFLoader-->	A PDF file
#CSVLoader-->	A CSV (spreadsheet) file
#WebBaseLoader-->	A website URL

#pip install pypdf


documents = loader.load()
print("Number of documents:", len(documents))
print("Content:")
print(documents[0].page_content)


#Key points to remember
#Data Ingestion = loading your data into the program.
#It is the first step of a RAG app.
#Use Document Loaders like TextLoader, PyPDFLoader, CSVLoader.
#.load() returns a list of Documents.
#Each document has .page_content (the actual text).



#2) Text Splitters
    #Text Splitters are used to split large documents into smaller chunks for processing.   
    #e.g:

#Two important settings
#chunk_size → how big each chunk is (number of characters).
#chunk_overlap → how much text repeats between chunks (so we don't lose meaning at the edges).

#A B C D E F G 
#[ A B C ] [ B C D ] [ D E F ] [E F G] [F G]

long_text = """Python is a programming language. It is easy to learn.
Streamlit is used to build web apps. LangChain helps build AI apps.
RAG combines search with AI models. Embeddings turn text into numbers.
Vector stores save those numbers. Retrievers find the best matches."""

print("Total characters:", len(long_text))

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,      # each chunk is about 80 characters
    chunk_overlap=10    # 10 characters repeat between chunks
)

chunks = splitter.split_text(long_text)

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)


#Splitting Documents (from a loader)

from langchain_community.document_loaders import TextLoader

documents = TextLoader("sample.txt").load()
chunks = splitter.split_documents(documents)
print("Number of chunks from documents:", len(chunks))


#3. Embeddings
#  Embeddings turn text into numbers (a list of numbers called a vector).

#Two ways to make embeddings
#Way	Model	Cost	API key?
#Free / local	HuggingFace all-MiniLM-L6-v2	Free	No
#OpenAI (paid)	text-embedding-3-small	Paid	Yes
#We will show both. They work almost the same way — only the model line changes.



from langchain_huggingface import HuggingFaceEmbeddings
# pip install langchain-huggingface sentence-transformers

# A small, free, popular embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding model is ready!")


text = "I love programming in Python."
vector = embeddings.embed_query(text)
print("How many numbers in this vector?", len(vector))
print("First 5 numbers:", vector[:5])


#Compare meanings
#Let's see that similar sentences get similar vectors.
import numpy as np

def similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = embeddings.embed_query("I like dogs")
v2 = embeddings.embed_query("I love puppies")
v3 = embeddings.embed_query("The car is fast")

print("dogs vs puppies (similar):", round(similarity(v1, v2), 2))
print("dogs vs car   (different):", round(similarity(v1, v3), 2))


#Key points to remember
    #Embeddings convert text into a list of numbers (a vector).
    #Similar meanings → similar numbers.
    #It is the third step in a RAG app.
    #embed_query() turns one sentence into a vector.
    #We compare vectors with cosine similarity (closer to 1 = more similar).
    #Free option: HuggingFace all-MiniLM-L6-v2 (no API key).
    #Paid option: OpenAI text-embedding-3-small (needs OPENAI_API_KEY).
    #Only the model line changes — the rest of the code stays the same.

#4. Vector Store
#A Vector Store is a special database that saves embeddings (the number vectors).

#It lets us search by meaning very quickly.

#Step 1: Install a free vector store (FAISS)
from langchain_huggingface import HuggingFaceEmbeddings

# Our small knowledge base (a list of sentences)
#Step 2: Prepare some text and the embedding model
texts = [
    "Python is a programming language.",
    "Streamlit is used to build web apps.",
    "Dogs are friendly animals.",
    "Cats like to sleep a lot."
]

# Free, local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Ready!")


#Step 3: Store the text in FAISSS
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_texts(texts, embeddings)
print("All texts are stored in the vector store!")


#Step 4: Search by meaning
query = "Tell me about pets"
results = vector_store.similarity_search(query, k=2)   # k=2 means top 2 matches
print("Top matches for:", query)
for r in results:
    print("-", r.page_content)


query = "which language is used"
results = vector_store.similarity_search(query, k=1)   # k=1 means top 1 match
print("Top matches for:", query)
for r in results:
    print("-", r.page_content)

#Step 5: Save and load the store
vector_store.save_local("my_store")
# load it back later
vector_store = FAISS.load_local("my_store", embeddings, allow_dangerous_deserialization=True)


#5. Retriever
#A Retriever is the tool that fetches the most relevant text from the vector store when you ask a question.
#It is the final step that connects your data to an AI model.

    #A Vector Store is the storage (the library shelf).
    #A Retriever is the helper that goes and brings back the right books for your question.

#A retriever is created from a vector store, and it has one simple job: given a question, return the best matching chunks.

#Create a Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) 
print("Retriever is ready!")

#Ask a question
query = "what is use for building web apps?"
results = retriever.invoke(query)

print("Question:", query)
print("\nRelevant information found:")
for r in results:
    print("-", r.page_content)

#User question ➜ Retriever finds relevant text ➜ AI model reads it ➜ AI gives the answer


#Key points to remember
    #A Retriever fetches the most relevant text for a question.
    #It is created from a vector store with .as_retriever().
    #k controls how many results to return.
    #Use retriever.invoke(question) to get the matching chunks.
    #Retriever is the bridge between your data and the AI model in a RAG app.

#6. Chains 
#A Chain in LangChain joins steps together so they run one after another, like a pipeline.
#The output of one step automatically becomes the input of the next step.
#This is what lets you build real AI apps instead of writing lots of glue code by hand.

#Step 1: Install and set your key
#pip install langchain langchain-openai
#$env:OPENAI_API_KEY = "your-key-here"

#Step 2: Create the three pieces

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt: a template with a blank {topic} to fill in
prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")

# 2. Model: the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")

# 3. Output parser: turn the model's reply into plain text
output_parser = StrOutputParser()

print("All three pieces are ready!")


#Step 3: Join them into a chain
chain = prompt | model | output_parser
print("Chain is ready!")


#Step 4: Run the chain
answer = chain.invoke({"topic": "Python"})
print(answer)