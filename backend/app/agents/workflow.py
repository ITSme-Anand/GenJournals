from langgraph.graph import StateGraph, START, END
from app.agents.state import AgentState
from app.agents.nodes.planner import plannerNode
from app.agents.nodes.load_memory import loadMemoryNode
from app.agents.nodes.retriever import retrieverNode
from app.agents.nodes.responder import responderNode
from app.agents.nodes.therapist import therapistNode
from app.agents.nodes.update_memory import updateMemoryNode
from app.agents.nodes.planRouter import planRouter
from app.agents.nodes.load_memory_router import loadMemoryRouter
graph = StateGraph(AgentState)

graph.add_node('Planner', plannerNode)
graph.add_node('Retriever', retrieverNode)
graph.add_node('LoadMemory', loadMemoryNode)
graph.add_node('Therapist', therapistNode)
graph.add_node('Responder', responderNode)
graph.add_node('updateMemory', updateMemoryNode)

graph.add_edge(START, 'Planner')
graph.add_edge('Planner', 'LoadMemory')
graph.add_conditional_edges('Planner',planRouter, {
    'END': END,
    'Retriever': 'Retriever'
} )
graph.add_edge('Retriever', 'Therapist')
graph.add_edge('Therapist', 'updateMemory')
graph.add_edge('updateMemory', END)
graph.add_conditional_edges('LoadMemory', loadMemoryRouter)
graph.add_edge('Responder', 'updateMemory')

workflow = graph.compile()

