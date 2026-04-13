from typing import Literal
from app.agents.state import AgentState

def loadMemoryRouter(state:AgentState)->Literal["Therapist","Responder"]:
    if(state['useRetriever']=='True'):return "Therapist"
    else: return "Responder"

