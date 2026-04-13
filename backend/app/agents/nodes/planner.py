from app.agents.state import AgentState
from app.agents.llm import GroqModel
from json import loads
SYSTEM_PROMPT = '''
    You are a planning agent for a mental health assistance application.
    You are given with a user's query.

    YOUR ROLE:
    Analyse the given query and classify whether we need to use retriever agent or not. 
    The retriever agent retrieves technical knowledge of psychology and mental health
    based on the user's query from a knowledge base.
    the retriever agent must only be used if the user requires mental health assistance.
    if the user sends a query that doesn't require mental health assistance, then the retriever
    agent need not be used. 

    Output JSON format ONLY:
    {
        "retrieve": "True"|"False",
        "reasoning": "Why did you choose to retrieve or not retrieve?"
    }

'''
def plannerNode(state: AgentState):
    """
    Decides if the retriever agent needs to be used or not.
    """
    userQuery = state['userQuery']
    response =  GroqModel.invoke(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": userQuery}
        ]
    )
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        content = content.replace("json", "").strip()
    plan = loads(content)
    #print(plan)
    return {"useRetriever": plan['retrieve'], "plannerReasoning": plan['reasoning']}

