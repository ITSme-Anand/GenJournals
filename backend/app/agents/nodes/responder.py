from app.agents.state import AgentState
from app.agents.llm import GroqModel
SYSTEM_PROMPT = '''
    You are an empathetic responder.
    Given a user Query along with some context about the user, your job is to respond
    to the user's query.
    You may or may not use the given context to write your response. \n
'''
def responderNode(state: AgentState):
    userContext = state['longTermMemory']
    FINAL_SYSTEM_PROMPT = SYSTEM_PROMPT + userContext

    response = GroqModel.invoke(input=[
        {"role":"system", "content": FINAL_SYSTEM_PROMPT},
        {"role":"user", "content": state['userQuery']}
    ])

    return {"response": [response.content]}