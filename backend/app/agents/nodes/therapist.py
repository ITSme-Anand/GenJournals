from app.agents.llm import GroqModel
from app.agents.state import AgentState

SYSTEM_PROMPT = '''
    You are a trained therapist agent of a mental health assistant application.
    the user needs some mental health assistance. 

    given the user's query, respond to the user in a very empathetic way, showing
    kindness at the same time help the user if needed. 
    for your reference, the retriever agent has retrieved all the required technical
    knowledge on Psychology and mental health for you, so that you can answer in the most 
    professional way. 
    Remember you are a professional therapist.
    And here is the book of knowledge you need to answer the user's query:

'''
def therapistNode(state: AgentState):
    context = state['retrievedKnowledge']
    FINAL_SYSTEM_PROMPT = SYSTEM_PROMPT + context
    response = GroqModel.invoke(input=[
        {"role":"system","content":FINAL_SYSTEM_PROMPT},
        {"role":"user","content":state['userQuery']}
    ])
    return {"response": [response.content]}