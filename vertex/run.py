import gradio as gr
import threading
import uvicorn
from vertex.app import app  # Your FastAPI app

# Start the FastAPI app in a background thread
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=7860)

threading.Thread(target=run_api, daemon=True).start()

# Dummy Gradio interface — doesn't need real inputs
def vertex():
    return "Backend is running."

# Launch dummy Gradio to satisfy HF Spaces
gr.Interface(
    fn=vertex,
    inputs=[],           # no inputs required
    outputs="text",
    title="Vertex API Wrapper"
).launch(server_name="0.0.0.0", server_port=7861)

