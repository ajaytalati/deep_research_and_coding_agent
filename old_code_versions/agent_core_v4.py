# agent_core_v4.py
# Implements TDS v5.1: Adds the Design Specification Agent.

import json
import re
import logging
import datetime
from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

logger = logging.getLogger("ResearchAgentV4")

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
    a2a_output: Optional[Dict]   # Stores the research_synthesis.json content
    
    # --- NEW KEY for v4 implementation (TDS v5.1) ---
    design_spec: Optional[str]   # Stores the generated design_spec.md content

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
        logger.info("Feedback detected, creating context for re-planning.")
        eval_report_str = json.dumps(state.get('evaluation_report'), indent=2)
        user_feedback_str = state.get('user_feedback', "No additional user feedback provided.")
        feedback_prefix = f"You are creating a new plan to revise a previous attempt. You MUST address the critiques from the last evaluation.\n### PREVIOUS EVALUATION REPORT:\n{eval_report_str}\n### USER'S ADDITIONAL FEEDBACK:\n{user_feedback_str}\n---\n"
    
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    base_prompt = prompts['create_plan_prompt'].format(context=context_str, user_prompt=state['user_prompt'])
    final_prompt = feedback_prefix + base_prompt
    response_content = llm.invoke(final_prompt).content
    
    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
    if json_match:
        try:
            state['plan'] = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            state['plan'] = {"error": "Failed to generate a valid plan."}
    else:
        state['plan'] = {"error": "No JSON object found in the LLM response for the plan."}
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
        
    logger.debug(f"Node: execute_synthesis_step for section {current_item['section_id']}")
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    plan_str = json.dumps(plan, indent=2)
    prompt = prompts['synthesis_step_prompt'].format(context=context_str, plan_str=plan_str, item_title=current_item['title'], item_description=current_item['description'])
    generated_content = llm.invoke(prompt).content
    state['working_memory'][current_item['section_id']] = generated_content
    state['completed_plan_items'].append(current_item['section_id'])
    return state

def assemble_and_finalize_draft(state: GraphState, llm, prompts):
    logger.debug("Node: assemble_and_finalize_draft")
    plan_items = state.get('plan', {}).get('plan_items', [])
    ordered_sections = sorted(plan_items, key=lambda x: str(x.get('section_id', '')))
    full_draft = "\n\n---\n\n".join([f"## {item['title']}\n\n{state['working_memory'].get(item['section_id'], '')}" for item in ordered_sections])
    
    citation_pattern = r'(\[Source: [^\]]+\])'
    citation_counter = 0
    def replace_with_index(match):
        nonlocal citation_counter
        citation_counter += 1
        return f"{match.group(1)}[{citation_counter}]"
    indexed_draft = re.sub(citation_pattern, replace_with_index, full_draft)
    state['output'] = indexed_draft

    logger.debug("Generating A2A synthesis artifact...")
    a2a_prompt = prompts['create_a2a_synthesis_prompt'].format(
        final_report_text=indexed_draft,
        current_date=datetime.date.today().isoformat()
    )
    response = llm.invoke(a2a_prompt)
    json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
    if json_match:
        try:
            state['a2a_output'] = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            state['a2a_output'] = {"error": "Failed to generate valid A2A JSON from LLM."}
    else:
        state['a2a_output'] = {"error": "No JSON object found in A2A response."}
    return state

def build_synthesis_graph(llm, prompts):
    graph = StateGraph(GraphState)
    graph.add_node("execute_synthesis_step", lambda s: execute_synthesis_step(s, llm, prompts))
    graph.add_node("assemble_and_finalize_draft", lambda s: assemble_and_finalize_draft(s, llm, prompts))
    def synthesis_router(state):
        plan, completed_count = state.get('plan', {}), len(state.get('completed_plan_items', []))
        return "assemble_and_finalize_draft" if completed_count >= len(plan.get('plan_items', [])) else "execute_synthesis_step"
    graph.set_entry_point("execute_synthesis_step")
    graph.add_conditional_edges("execute_synthesis_step", synthesis_router)
    graph.add_edge("assemble_and_finalize_draft", END)
    return graph.compile()

# --- EVALUATION GRAPH ---

def generate_evaluation_node(state: GraphState, llm, prompts):
    logger.debug("--- Entering Evaluation Graph ---")
    prompt = prompts['evaluation_prompt_v2'].format(user_prompt=state['user_prompt'], output=state['output'])
    response = llm.invoke(prompt)
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            state['evaluation_report'] = json.loads(json_match.group(0))
        else:
            raise json.JSONDecodeError("No JSON object found", response.content, 0)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse final evaluation JSON: {e}")
        state['evaluation_report'] = {"error": "Failed to generate valid final evaluation JSON."}
    return state

def build_evaluation_graph(llm, prompts, tools):
    model_with_tools = llm.bind_tools(tools)
    graph = StateGraph(GraphState)
    graph.add_node("generate_evaluation", lambda s: generate_evaluation_node(s, model_with_tools, prompts))
    graph.set_entry_point("generate_evaluation")
    graph.add_edge("generate_evaluation", END)
    return graph.compile()

# --- DESIGN AGENT GRAPH (TDS v5.1) ---

def generate_design_spec_node(state: GraphState, llm, prompts):
    """
    Node: Generates a technical design specification from the research_synthesis.json artifact.
    """
    logger.debug("--- Entering Design Agent Graph ---")
    logger.debug("Node: generate_design_spec_node")
    
    research_synthesis = state.get('a2a_output')
    if not research_synthesis or 'error' in research_synthesis:
        logger.error("Cannot generate design spec: research_synthesis.json is missing or contains an error.")
        state['design_spec'] = "# ERROR: Could not generate Design Specification because the Research Synthesis artifact is missing or invalid."
        return state

    prompt = prompts['create_design_spec_prompt'].format(
        research_synthesis_json=json.dumps(research_synthesis, indent=2)
    )
    
    response = llm.invoke(prompt)
    state['design_spec'] = response.content
    logger.debug("Successfully generated design specification.")
    return state

def build_design_agent_graph(llm, prompts):
    """
    Factory function to build the Design Specification Agent graph.
    """
    graph = StateGraph(GraphState)
    graph.add_node("generate_design_spec", lambda s: generate_design_spec_node(s, llm, prompts))
    graph.set_entry_point("generate_design_spec")
    graph.add_edge("generate_design_spec", END)
    return graph.compile()

