import chromadb
from chromadb import PersistentClient  # Add this import
from sentence_transformers import SentenceTransformer
import ollama
import os

# --- IMPORTANT FIX ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(script_dir)
# --- END OF FIX ---

# --- 1. Connect to our databases and model ---
try:
    # Use PersistentClient to correctly connect to the saved database
    vector_client = PersistentClient(path="./chroma")
    collection = vector_client.get_collection("argo_summaries")
except Exception as e:
    print(f"❌ Error connecting to ChromaDB: {e}")
    print("Please ensure your 'chroma' folder is in the same directory and contains the 'argo_summaries' collection.")
    exit()

# Load the same embedding model used for data ingestion
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. Define the RAG Pipeline ---
def retrieve_context(query_text, n_results=5):
    # ... (rest of the function is the same)
    print("⏳ Retrieving context from vector database...")
    query_embedding = embedding_model.encode([query_text]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    context = results['documents'][0]
    print("✅ Context retrieved.")
    return context

def generate_response_with_llm(context, user_query):
    # ... (rest of the function is the same)
    print("⏳ Generating response with LLM...")
    
    prompt = f"""
    You are an expert oceanographer. Use the following context to answer the user's question.
    If the context does not contain the answer, say that you cannot provide information on that topic.
    
    Context:
    {context}
    
    Question:
    {user_query}
    
    Answer:
    """
    
    try:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        answer = response['message']['content']
        print("✅ Response generated.")
        return answer
    except Exception as e:
        return f"❌ Error communicating with Ollama: {e}. Please ensure Ollama is running and the 'llama3' model is downloaded."

# --- 3. Main function to tie it all together ---
def get_answer(user_query):
    # ... (rest of the function is the same)
    context = retrieve_context(user_query)
    answer = generate_response_with_llm(context, user_query)
    
    return answer

# Example usage (for testing)
if __name__ == "__main__":
    test_query = "What were the temperature and salinity measurements for float 5906142?"
    print(f"User Query: {test_query}")
    response = get_answer(test_query)
    print("\n------------------")
    print("Final Answer:")
    print(response)
    print("------------------\n")