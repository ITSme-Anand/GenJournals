from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    userQuery: str #represents the query of user
    useRetriever: bool
    retrievedKnowledge:str
    longTermMemory: str
    response: str
    messages: Annotated[list[BaseMessage], add_messages]