from app.agents.state import AgentState

def retrieverNode(state: AgentState):
    #here we have to deal with retrieval of knowledge
    print('retriever is called!')
    return {"retrievedKnowledge": "eating biriyani causes obesity, tiredness, fatigue and sometimes even cancer"}