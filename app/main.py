from fastapi import FastAPI
from pydantic import BaseModel
from . import logic # Import our logic file

# Create the FastAPI app instance
app = FastAPI(title="NLU Service")

# Define the structure of the incoming request body
class NLURequest(BaseModel):
    text: str
    conversation_id: str = "default" # Optional ID to track conversations

# This is a special event handler that runs when the server starts up.
# It's the perfect place to load our heavy NLP model once.
@app.on_event("startup")
def startup_event():
    logic.initialize_nlp()

# Define our main API endpoint
@app.post("/parse")
def parse_text(request: NLURequest):
    """
    Receives text and returns the structured NLU analysis.
    """
    # Call the main processing function from our logic module
    structured_data = logic.process_text(request.text)
    return structured_data

# A simple root endpoint to check if the server is running
@app.get("/")
def read_root():
    return {"status": "NLU Service is running."}