# Argo Ocean Data Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Argo oceanographic data. It uses a local vector database (ChromaDB) to provide a Large Language Model (LLM) with specific, relevant context.

## Project Structure

- `convert_netcdf_to_sql.py`: Converts raw NetCDF (.nc) files into a structured SQLite database (`argo.db`).
- `convert_sql_to_vector.py`: Reads data from the SQL database, generates embeddings, and stores them in the local vector database (`chroma` folder).
- `rag_pipeline.py`: Contains the core RAG logic. It retrieves relevant data from the vector database and sends it to the LLM for a final response.
- `chatbot_interface.py`: A Gradio-based web interface for interacting with the chatbot.
- `requirements.txt`: Lists all Python libraries needed to run the project.

## How to Set Up and Run the Project

Follow these steps to get the chatbot running on your local machine:

### Step 1: Clone the Repository

Clone this project to your local machine using Git:

```bash
git clone [https://github.com/abhishektayde15/argo-chatbot-rag.git](https://github.com/abhishektayde15/argo-chatbot-rag.git)
cd argo-chatbot-rag
