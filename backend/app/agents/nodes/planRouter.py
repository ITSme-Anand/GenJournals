from typing import Literal
from app.agents.state import AgentState

def planRouter(state:AgentState)->Literal['Retriever','END']:
    print('inside planRouter')
    if(state['useRetriever']=='True'):
        print('going to return Retriever')
        return "Retriever"
    else: return "END"