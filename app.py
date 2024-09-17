import time
import streamlit as st
import cohere
import pinecone
import PyPDF2
from st_copy_to_clipboard import st_copy_to_clipboard

# Initialize Cohere client
co = cohere.Client(st.secrets["COHERE_API_KEY"])

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "quickstart"
index = pc.Index(index_name)


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def process_document(text):
    # Split the text into chunks (you may need to implement a more sophisticated chunking method)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    # Generate embeddings
    embeds = co.embed(
        texts=chunks,
        model='embed-english-v3.0',
        input_type='search_document',
        truncate='END'
    ).embeddings

    # Upsert to Pinecone
    to_upsert = [(str(i), embed, {"chunk": chunk})
                 for i, (embed, chunk) in enumerate(zip(embeds, chunks))]
    index.upsert(vectors=to_upsert)


def get_answer(query):
    # Generate query embedding
    xq = co.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query',
        truncate='END'
    ).embeddings

    # Query Pinecone
    res = index.query(vector=xq[0], top_k=5, include_metadata=True)

    # Print the structure of 'res' to verify its contents (for debugging)
    # This will help you see if the structure is as expected
    print("Query Results:", res)

    # Prepare context
    context = ""
    for match in res['matches']:
        # Check if 'metadata' exists and contains 'chunk'
        if 'metadata' in match and 'chunk' in match['metadata']:
            context += f"{match['metadata']['chunk']}\n\n"

    # Generate answer using Cohere
    response = co.chat(
        message=f"""
        You are a specialized RAG Bot and your name is RAGSTA which is developed by Vigneswaran, and your primary function is to answer queries based on files uploaded by the user. 
        You only process and respond using information from these files and you will not answer any questions outside of the context of the files uploaded by the user. If working principle is asked, answer with following info: My working principle is based on RAG (Retrieval-Augmented Generation) where you search for the information in the files uploaded by the user.

        User Query: {query}
        Relevant Context:
        {context}
        """,
        model="command-r-plus"
    )

    return response.text, context


# Main App
st.set_page_config(
    page_title="RAGSTA Bot",  # Custom page title
    page_icon="bot.png",  # Path to the favicon
    layout="wide"
)

st.title("RAGSTA Bot 🤖 - Developed by Vigneswaran")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    process_document(text)
    st.success("Document processed and indexed successfully!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the pdf:")

if query:

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.status("Searching...", expanded=True) as status:
        answer, context = get_answer(query)
        for chunk in context.split('\n\n'):
            if chunk:
                st.write(chunk)
        status.update(
            label="Extracting info...", state="running", expanded=True
        )
        time.sleep(3)
        status.update(
            label="Found the Citations!", state="complete", expanded=False
        )

    with st.chat_message("assistant"):
        st.write(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Add a "Copy Answer" button next to the answer
    st_copy_to_clipboard(answer)
    # Display assistant response in chat message container


st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a PDF document using the file uploader.
2. Wait for the document to be processed and indexed.
3. Enter your question in the text input field.
4. View the answer and retrieved document segments.
""")
