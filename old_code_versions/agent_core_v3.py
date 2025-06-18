# agent_core_v3.py
# Implements TDS v5.0: A2A (Agent-to-Agent) Artifact Generation.

import json
import re
import logging
import datetime
from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# Use the same logger name as the supervisor for consistent logging
logger = logging.getLogger("ResearchAgentV3")

class GraphState(TypedDict):
    """
    Defines the central state object for the agentic graphs.
    """
    user_prompt: str
    documents: List[Dict]
    plan: Optional[Dict]
    output: str                  # Stores the human-readable Markdown report
    evaluation_report: Optional[Dict]
    user_feedback: Optional[str]
    messages: List[BaseMessage]
    working_memory: Optional[Dict]
    completed_plan_items: List[str]
    
    # --- NEW KEY for v3 implementation (TDS v5.0) ---
    a2a_output: Optional[Dict] # Will hold the structured research_synthesis.json content

# --- PLANNING GRAPH ---

def retrieve_context(state: GraphState, retriever):
    """
    Node: Retrieves documents from the vector store based on the user's prompt.
    """
    logger.debug("Node: retrieve_context")
    docs = retriever.get_relevant_documents(state["user_prompt"])
    state['documents'] = [{"name": doc.metadata.get('source', 'unknown'), "content": doc.page_content} for doc in docs]
    logger.debug(f"Retrieved {len(state['documents'])} documents.")
    return state

def generate_plan(state: GraphState, llm, prompts):
    """
    Node: Generates a structured JSON plan for the report.
    Handles initial planning and re-planning based on feedback.
    """
    logger.debug("Node: generate_plan")
    
    # Dynamically create a prefix for the prompt if feedback exists
    feedback_prefix = ""
    if state.get('user_feedback') or state.get('evaluation_report'):
        logger.info("Feedback detected, creating context for re-planning.")
        eval_report_str = json.dumps(state.get('evaluation_report'), indent=2)
        user_feedback_str = state.get('user_feedback', "No additional user feedback provided.")
        
        feedback_prefix = f"""
You are creating a new plan to revise a previous attempt. You MUST address the critiques from the last evaluation.
### PREVIOUS EVALUATION REPORT:
{eval_report_str}

### USER'S ADDITIONAL FEEDBACK:
{user_feedback_str}

Based on this feedback, create a new and improved plan to address these issues.
---
"""
    
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    base_prompt = prompts['create_plan_prompt'].format(context=context_str, user_prompt=state['user_prompt'])
    
    final_prompt = feedback_prefix + base_prompt

    response_content = llm.invoke(final_prompt).content
    logger.debug(f"LLM Raw Plan Response:\n{response_content}")
    
    # Robustly parse JSON from the LLM's response
    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
    if json_match:
        json_string = json_match.group(0)
        try:
            plan_json = json.loads(json_string)
            state['plan'] = plan_json
            logger.debug("Successfully parsed plan JSON.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON from plan response: {e}")
            state['plan'] = {"error": "Failed to generate a valid plan."}
    else:
        logger.error("No JSON object found in the LLM response for the plan.")
        state['plan'] = {"error": "Failed to generate a valid plan."}
        
    return state

def build_planning_graph(llm, retriever, prompts):
    """Factory function to build the planning graph."""
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", lambda s: retrieve_context(s, retriever))
    graph.add_node("generate_plan", lambda s: generate_plan(s, llm, prompts))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate_plan")
    graph.add_edge("generate_plan", END)
    return graph.compile()


# --- SYNTHESIS GRAPH ---

def execute_synthesis_step(state: GraphState, llm, prompts):
    """
    Node: Generates content for a single section of the plan in a loop.
    """
    plan, completed = state.get('plan', {}), state.get('completed_plan_items', [])
    current_item = next((item for item in plan.get('plan_items', []) if item['section_id'] not in completed), None)
    
    if not current_item:
        logger.warning("Synthesis step called but no more items to process.")
        return state
        
    logger.debug(f"Node: execute_synthesis_step for section {current_item['section_id']}")
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    plan_str = json.dumps(plan, indent=2)
    prompt = prompts['synthesis_step_prompt'].format(context=context_str, plan_str=plan_str, item_title=current_item['title'], item_description=current_item['description'])
    
    generated_content = llm.invoke(prompt).content
    state['working_memory'][current_item['section_id']] = generated_content
    state['completed_plan_items'].append(current_item['section_id'])
    return state

def assemble_and_finalize_draft(state: GraphState, llm, prompts):
    """
    Node: Assembles the final human-readable report AND generates the machine-readable A2A artifact.
    """
    logger.debug("Node: assemble_and_finalize_draft")
    
    # 1. Assemble the human-readable Markdown report
    plan_items = state.get('plan', {}).get('plan_items', [])
    ordered_sections = sorted(plan_items, key=lambda x: str(x.get('section_id', '')))
    full_draft = "\n\n---\n\n".join([f"## {item['title']}\n\n{state['working_memory'].get(item['section_id'], '')}" for item in ordered_sections])
    
    # Add numeric indices to citations for traceability
    logger.debug("Adding numeric indices to citations...")
    citation_pattern = r'(\[Source: [^\]]+\])'
    citation_counter = 0
    def replace_with_index(match):
        nonlocal citation_counter
        citation_counter += 1
        return f"{match.group(1)}[{citation_counter}]"
    indexed_draft = re.sub(citation_pattern, replace_with_index, full_draft)
    state['output'] = indexed_draft
    logger.debug("Human-readable report assembly complete.")

    # 2. Generate the Agent-to-Agent (A2A) JSON artifact (TDS v5.0)
    logger.debug("Generating A2A synthesis artifact...")
    a2a_prompt = prompts['create_a2a_synthesis_prompt'].format(final_report_text=indexed_draft)
    
    # Add the current date to the prompt for the schema
    a2a_prompt = a2a_prompt.replace("<The current date in YYYY-MM-DD format>", datetime.date.today().isoformat())

    response = llm.invoke(a2a_prompt)
    logger.debug(f"LLM Raw A2A Response:\n{response.content}")
    
    # Robustly parse JSON from the A2A response
    json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
    if json_match:
        try:
            a2a_json = json.loads(json_match.group(0))
            state['a2a_output'] = a2a_json
            logger.debug("Successfully parsed A2A synthesis JSON.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse A2A JSON response: {e}")
            state['a2a_output'] = {"error": "Failed to generate valid A2A JSON from LLM."}
    else:
        logger.error("No JSON object found in A2A response.")
        state['a2a_output'] = {"error": "No JSON object found in A2A response."}

    return state

def build_synthesis_graph(llm, prompts):
    """Factory function to build the synthesis graph."""
    graph = StateGraph(GraphState)
    graph.add_node("execute_synthesis_step", lambda s: execute_synthesis_step(s, llm, prompts))
    # --- MODIFICATION: Pass llm and prompts to the assembly node ---
    graph.add_node("assemble_and_finalize_draft", lambda s: assemble_and_finalize_draft(s, llm, prompts))

    def synthesis_router(state):
        plan, completed_count = state.get('plan', {}), len(state.get('completed_plan_items', []))
        total_count = len(plan.get('plan_items', []))
        return "assemble_and_finalize_draft" if completed_count >= total_count else "execute_synthesis_step"

    graph.set_entry_point("execute_synthesis_step")
    graph.add_conditional_edges("execute_synthesis_step", synthesis_router)
    graph.add_edge("assemble_and_finalize_draft", END)
    return graph.compile()


# --- EVALUATION GRAPH ---

def generate_evaluation_node(state: GraphState, llm, prompts):
    """
    Node: Performs the entire audit process in a single, tool-augmented call for robustness.
    """
    logger.debug("--- Entering Evaluation Graph ---")
    logger.debug("Node: generate_evaluation_node")
    prompt = prompts['evaluation_prompt_v2'].format(user_prompt=state['user_prompt'], output=state['output'])
    logger.debug(f"PROMPT FOR EVALUATOR:\n{prompt}")
    
    response = llm.invoke(prompt)
    logger.debug(f"RAW RESPONSE FROM EVALUATOR:\n{response.content}")
    
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            eval_json = json.loads(json_match.group(0))
            state['evaluation_report'] = eval_json
            logger.debug(f"Successfully parsed final evaluation JSON:\n{json.dumps(eval_json, indent=2)}")
        else:
            raise json.JSONDecodeError("No JSON object found in the final LLM response.", response.content, 0)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse final evaluation JSON: {e}")
        state['evaluation_report'] = {"error": "Failed to generate valid final evaluation JSON from LLM response."}
        
    return state

def build_evaluation_graph(llm, prompts, tools):
    """Factory function to build the evaluation graph."""
    # Bind the tools to the LLM so it can use them
    model_with_tools = llm.bind_tools(tools)
    
    graph = StateGraph(GraphState)
    graph.add_node("generate_evaluation", lambda s: generate_evaluation_node(s, model_with_tools, prompts))
    graph.set_entry_point("generate_evaluation")
    graph.add_edge("generate_evaluation", END)
    return graph.compile()

