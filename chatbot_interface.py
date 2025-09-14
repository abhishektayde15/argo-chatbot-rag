import gradio as gr
from rag_pipeline import get_answer

# Define the Gradio interface
iface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the Argo data..."),
    outputs="text",
    title="Argo Ocean Data Chatbot",
    description="Ask a question about the ocean data from the Argo profiles. For example: 'What were the temperatures near the equator in January?'"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()