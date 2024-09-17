# RAGSTA Bot 🤖

## Overview 🔥

RAGSTA Bot is a specialized RAG (Retrieval-Augmented Generation) chatbot developed by Vigneswaran. It's designed to answer queries based on user-uploaded PDF documents. The bot processes the uploaded documents, indexes their content, and uses this information to provide accurate and context-specific responses to user queries by using services like Cohere and Pinecone.

## Accessing the Live Application 🛜

You can access the live RAGSTA Bot application at: https://ragsta.streamlit.app/

## Key Features ✅

- PDF document upload and processing
- Text extraction and indexing
- Query-based information retrieval
- Context-aware response generation
- User-friendly chat interface

## How It Works ⚙️

1. **Document Processing**: When a user uploads a PDF, the bot extracts the text and splits it into manageable chunks.
2. **Embedding Generation**: Each chunk is converted into a vector embedding using Cohere's embedding model.
3. **Indexing**: The embeddings are stored in a Pinecone vector database for efficient retrieval.
4. **Query Processing**: User queries are also converted to embeddings and used to search the Pinecone index.
5. **Context Retrieval**: The most relevant document chunks are retrieved based on the query.
6. **Response Generation**: Cohere's language model generates a response using the retrieved context and the user's query.

## Setup and Installation 🗒️

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Vigneswaran321/RAGSTA.git
   cd ragsta-bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.streamlit/secrets.toml` file with your API keys:
   ```
   COHERE_API_KEY = "your_cohere_api_key"
   PINECONE_API_KEY = "your_pinecone_api_key"
   ```

### Running the Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t ragsta-bot .
   ```

2. Run the container:
   ```
   docker run -p 8501:8501 ragsta-bot
   ```

## Project Structure

```
ragsta-bot/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── .streamlit/
│   └── secrets.toml       # API keys and secrets (not in version control)
├── bot.png                # Bot icon for the web interface
├── README.md              # Project documentation
└── Notebook/
    ├── dataset.csv        # Dataset file
    └── ragbot.ipynb       # Jupyter notebook for development and testing
```


## Usage 😎

1. Upload a PDF document using the file uploader on the main page.
2. Wait for the document to be processed and indexed.
3. Enter your question in the chat input field at the bottom of the page.
4. View the bot's response and the relevant context extracted from the document.
5. Use the "Copy Answer" button to easily copy the bot's response.

## Technologies Used 🧑‍💻

- Streamlit: For creating the web interface
- Cohere: For generating text embeddings and responses
- Pinecone: For indexing and querying document embeddings
- PyPDF2: For extracting text from PDF files

## Contributing 👇

Contributions to RAGSTA Bot are welcome! Please feel free to submit a Pull Request.


## Contact 📞

For any queries or support, please contact Vigneswaran at https://vigneswaran.framer.ai/.
