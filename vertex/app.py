from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import assess_risk
import uvicorn

app = FastAPI()

origins = [
    "https://etimbukafia.github.io",            
]

# Allow requests from your GitHub Pages domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/assess")
def generate_assessment():
    """
    Endpoint to generate a risk assessment report.
    """
    # Call the main function from vertex.main
    result = assess_risk()
    
    # Return the result as JSON
    return {"result": result}

