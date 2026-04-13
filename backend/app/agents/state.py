from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    userQuery: str #represents the query of user
    useRetriever: str
    plannerReasoning: str
    retrievedKnowledge:str
    longTermMemory: str
    updatedMemory: str
    response: Annotated[list[BaseMessage], add_messages]
    messages: Annotated[list[BaseMessage], add_messages]