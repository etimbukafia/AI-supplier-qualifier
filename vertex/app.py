from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vertex.main import main

app = FastAPI()

# Allow requests from your GitHub Pages domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://etimbukafia.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/assess")
def generate_assessment():
    """
    Endpoint to generate a risk assessment report.
    """
    # Call the main function from vertex.main
    result = main()
    
    # Return the result as JSON
    return {"result": result}
