# agent_core_v5.py
# Implements TDS v6.0: Adds generation of design_synthesis.json.

import json
import re
import logging
import datetime
from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

logger = logging.getLogger("ResearchAgentV5")

class GraphState(TypedDict):
    """
    Defines the central state object for the agentic graphs.
    """
    user_prompt: str
    documents: List[Dict]
    plan: Optional[Dict]
    output: str                  # Stores the human-readable Markdown research report
    evaluation_report: Optional[Dict]
    user_feedback: Optional[str]
    messages: List[BaseMessage]
    working_memory: Optional[Dict]
    completed_plan_items: List[str]
    a2a_output: Optional[Dict]   # Stores research_synthesis.json content
    design_spec: Optional[str]   # Stores design_spec.md content
    
    # --- NEW KEY for v5 implementation (TDS v6.0) ---
    design_synthesis_json: Optional[Dict] # Will hold the design work order

# --- PLANNING GRAPH ---
def retrieve_context(state: GraphState, retriever):
    logger.debug("Node: retrieve_context")
    docs = retriever.get_relevant_documents(state["user_prompt"])
    state['documents'] = [{"name": doc.metadata.get('source', 'unknown'), "content": doc.page_content} for doc in docs]
    return state

def generate_plan(state: GraphState, llm, prompts):
    logger.debug("Node: generate_plan")
    feedback_prefix = ""
    if state.get('user_feedback') or state.get('evaluation_report'):
        eval_report_str = json.dumps(state.get('evaluation_report'), indent=2)
        user_feedback_str = state.get('user_feedback', "No additional user feedback provided.")
        feedback_prefix = f"You are revising a plan. Address these critiques:\nEVALUATION:\n{eval_report_str}\n\nUSER FEEDBACK:\n{user_feedback_str}\n---\n"
    
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    prompt = prompts['create_plan_prompt'].format(context=context_str, user_prompt=state['user_prompt'])
    final_prompt = feedback_prefix + prompt
    response_content = llm.invoke(final_prompt).content
    
    try:
        state['plan'] = json.loads(re.search(r'\{.*\}', response_content, re.DOTALL).group(0))
    except (json.JSONDecodeError, AttributeError):
        state['plan'] = {"error": "Failed to generate a valid plan JSON."}
    return state

def build_planning_graph(llm, retriever, prompts):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", lambda s: retrieve_context(s, retriever))
    graph.add_node("generate_plan", lambda s: generate_plan(s, llm, prompts))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate_plan")
    graph.add_edge("generate_plan", END)
    return graph.compile()

# --- SYNTHESIS GRAPH ---
def execute_synthesis_step(state: GraphState, llm, prompts):
    plan, completed = state.get('plan', {}), state.get('completed_plan_items', [])
    current_item = next((item for item in plan.get('plan_items', []) if item['section_id'] not in completed), None)
    if not current_item: return state
        
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    plan_str = json.dumps(plan, indent=2)
    prompt = prompts['synthesis_step_prompt'].format(context=context_str, plan_str=plan_str, item_title=current_item['title'], item_description=current_item['description'])
    state['working_memory'][current_item['section_id']] = llm.invoke(prompt).content
    state['completed_plan_items'].append(current_item['section_id'])
    return state

def assemble_and_finalize_draft(state: GraphState, llm, prompts):
    plan_items = state.get('plan', {}).get('plan_items', [])
    ordered_sections = sorted(plan_items, key=lambda x: str(x.get('section_id', '')))
    full_draft = "\n\n---\n\n".join([f"## {item['title']}\n\n{state['working_memory'].get(item['section_id'], '')}" for item in ordered_sections])
    
    citation_counter = 0
    def replace_with_index(match):
        nonlocal citation_counter
        citation_counter += 1
        return f"{match.group(1)}[{citation_counter}]"
    state['output'] = re.sub(r'(\[Source: [^\]]+\])', replace_with_index, full_draft)

    a2a_prompt = prompts['create_a2a_synthesis_prompt'].format(final_report_text=state['output'], current_date=datetime.date.today().isoformat())
    response = llm.invoke(a2a_prompt)
    try:
        state['a2a_output'] = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group(0))
    except (json.JSONDecodeError, AttributeError):
        state['a2a_output'] = {"error": "Failed to generate A2A synthesis JSON."}
    return state

def build_synthesis_graph(llm, prompts):
    graph = StateGraph(GraphState)
    graph.add_node("execute_synthesis_step", lambda s: execute_synthesis_step(s, llm, prompts))
    graph.add_node("assemble_and_finalize_draft", lambda s: assemble_and_finalize_draft(s, llm, prompts))
    def synthesis_router(state):
        plan, completed = state.get('plan', {}), state.get('completed_plan_items', [])
        return "assemble_and_finalize_draft" if len(completed) >= len(plan.get('plan_items', [])) else "execute_synthesis_step"
    graph.set_entry_point("execute_synthesis_step")
    graph.add_conditional_edges("execute_synthesis_step", synthesis_router)
    graph.add_edge("assemble_and_finalize_draft", END)
    return graph.compile()

# --- EVALUATION GRAPH ---
def generate_evaluation_node(state: GraphState, llm, prompts):
    prompt = prompts['evaluation_prompt_v2'].format(user_prompt=state['user_prompt'], output=state['output'])
    response = llm.invoke(prompt)
    try:
        state['evaluation_report'] = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group(0))
    except (json.JSONDecodeError, AttributeError):
        state['evaluation_report'] = {"error": "Failed to generate evaluation JSON."}
    return state

def build_evaluation_graph(llm, prompts, tools):
    model_with_tools = llm.bind_tools(tools)
    graph = StateGraph(GraphState)
    graph.add_node("generate_evaluation", lambda s: generate_evaluation_node(s, model_with_tools, prompts))
    graph.set_entry_point("generate_evaluation")
    graph.add_edge("generate_evaluation", END)
    return graph.compile()

# --- DESIGN AGENT GRAPH ---
def generate_design_spec_node(state: GraphState, llm, prompts):
    logger.debug("--- Entering Design Agent Graph ---")
    research_synthesis = state.get('a2a_output')
    if not research_synthesis or 'error' in research_synthesis:
        state['design_spec'] = "# ERROR: Research Synthesis artifact is missing or invalid."
        state['design_synthesis_json'] = {"error": "Upstream artifact missing."}
        return state

    # 1. Generate Markdown Design Spec
    md_prompt = prompts['create_design_spec_prompt'].format(research_synthesis_json=json.dumps(research_synthesis, indent=2))
    state['design_spec'] = llm.invoke(md_prompt).content
    logger.debug("Successfully generated markdown design specification.")

    # 2. Generate JSON Design Synthesis from the Markdown Spec (TDS v6.0)
    json_prompt = prompts['create_design_synthesis_prompt'].format(
        design_spec_md=state['design_spec'],
        current_date=datetime.date.today().isoformat()
    )
    response = llm.invoke(json_prompt)
    try:
        state['design_synthesis_json'] = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group(0))
        logger.debug("Successfully generated JSON design synthesis.")
    except (json.JSONDecodeError, AttributeError):
        state['design_synthesis_json'] = {"error": "Failed to generate design synthesis JSON."}
        logger.error("Failed to parse design synthesis JSON from LLM response.")
    
    return state

def build_design_agent_graph(llm, prompts):
    graph = StateGraph(GraphState)
    graph.add_node("generate_design_spec", lambda s: generate_design_spec_node(s, llm, prompts))
    graph.set_entry_point("generate_design_spec")
    graph.add_edge("generate_design_spec", END)
    return graph.compile()

