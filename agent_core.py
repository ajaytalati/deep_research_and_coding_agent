# agent_core.py
# v3.7: Rewrote finalize_evaluation_json for robustness and added detailed logging.

import json
import re
import logging
from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- State Definition ---
class GraphState(TypedDict):
    user_prompt: str
    documents: List[Dict]
    plan: Optional[Dict]
    output: str
    evaluation_report: Optional[Dict]
    user_feedback: Optional[str]
    log: List[str]
    messages: List[BaseMessage]
    working_memory: Optional[Dict]
    completed_plan_items: List[str]
    parsed_claims: List[Dict]

# --- (Functions from retrieve_context to build_synthesis_graph are unchanged) ---
def retrieve_context(state: GraphState, retriever):
    logging.debug("Node: retrieve_context")
    docs = retriever.get_relevant_documents(state["user_prompt"])
    state['documents'] = [{"name": doc.metadata.get('source', 'unknown'), "content": doc.page_content} for doc in docs]
    logging.debug(f"Retrieved {len(state['documents'])} documents.")
    return state

def generate_plan(state: GraphState, llm, prompts):
    logging.debug("Node: generate_plan")
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    prompt = prompts['create_plan_prompt'].format(context=context_str, user_prompt=state['user_prompt'])
    response_content = llm.invoke(prompt).content
    logging.debug(f"LLM Raw Plan Response:\n{response_content}")
    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
    if json_match:
        json_string = json_match.group(0)
        try:
            plan_json = json.loads(json_string)
            state['plan'] = plan_json
            logging.debug("Successfully parsed plan JSON.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse extracted JSON: {e}")
            state['plan'] = {"error": "Failed to generate a valid plan."}
    else:
        logging.error("No JSON object found in the LLM response for the plan.")
        state['plan'] = {"error": "Failed to generate a valid plan."}
    return state

def build_planning_graph(llm, retriever, prompts):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", lambda s: retrieve_context(s, retriever))
    graph.add_node("generate_plan", lambda s: generate_plan(s, llm, prompts))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate_plan")
    graph.add_edge("generate_plan", END)
    return graph.compile()

def execute_synthesis_step(state: GraphState, llm, prompts):
    plan = state.get('plan', {}); completed = state.get('completed_plan_items', [])
    current_item = next((item for item in plan.get('plan_items', []) if item['section_id'] not in completed), None)
    if not current_item: return state
    logging.debug(f"Node: execute_synthesis_step for section {current_item['section_id']}")
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    plan_str = json.dumps(plan, indent=2)
    prompt = prompts['synthesis_step_prompt'].format(context=context_str, plan_str=plan_str, item_title=current_item['title'], item_description=current_item['description'])
    generated_content = llm.invoke(prompt).content
    state['working_memory'][current_item['section_id']] = generated_content
    state['completed_plan_items'].append(current_item['section_id'])
    return state

def assemble_draft(state: GraphState):
    logging.debug("Node: assemble_draft")
    plan_items = state.get('plan', {}).get('plan_items', [])
    ordered_sections = sorted(plan_items, key=lambda x: x['section_id'])
    full_draft = [f"## {item['title']}\n\n{state['working_memory'].get(item['section_id'], '')}" for item in ordered_sections]
    state['output'] = "\n\n---\n\n".join(full_draft)
    return state

def build_synthesis_graph(llm, prompts):
    graph = StateGraph(GraphState)
    graph.add_node("execute_synthesis_step", lambda s: execute_synthesis_step(s, llm, prompts))
    graph.add_node("assemble_draft", assemble_draft)
    def synthesis_router(state):
        plan = state.get('plan', {}); completed_count = len(state.get('completed_plan_items', [])); total_count = len(plan.get('plan_items', []))
        return "assemble_draft" if completed_count >= total_count else "execute_synthesis_step"
    graph.set_entry_point("execute_synthesis_step")
    graph.add_conditional_edges("execute_synthesis_step", synthesis_router)
    graph.add_edge("assemble_draft", END)
    return graph.compile()


# --- Evaluation Graph (Teacher with Tools) ---

def parse_claims(state: GraphState):
    logging.debug(f"--- Entering Evaluation Graph ---")
    logging.debug(f"Node: parse_claims")
    text = state['output']
    claim_pattern = re.compile(r'([^\.!?]*?\[Source:[^\]]+\][\s\.]*)')
    claims = claim_pattern.findall(text)
    state['parsed_claims'] = [{"claim": claim.strip()} for claim in claims]
    logging.debug(f"Found {len(state['parsed_claims'])} claims to audit.")
    return state

def generate_evaluation(state: GraphState, llm, prompts):
    logging.debug("Node: generate_evaluation")
    prompt = prompts['evaluation_prompt'].format(user_prompt=state['user_prompt'], output=state['output'])
    logging.debug(f"Invoking evaluation LLM with tool-binding.")
    response_with_tool_calls = llm.invoke(prompt)
    logging.debug(f"LLM evaluation response received:\n{response_with_tool_calls}")
    return {"messages": [response_with_tool_calls]}

# --- FIX: New robust finalization function ---
def finalize_evaluation_json(state: GraphState, llm):
    logging.debug("Node: finalize_evaluation_json")
    
    # Check if we have any messages to process
    if not state.get('messages'):
        logging.error("CRITICAL: Reached finalize_evaluation_json with no messages in state.")
        state['evaluation_report'] = {"error": "Internal state error: No messages to process for final evaluation."}
        return state

    # Construct a new, clean prompt with all the information
    # This avoids the fragile llm.invoke(messages) call that was causing the error
    tool_results_str = ""
    for msg in state['messages']:
        if isinstance(msg, ToolMessage):
            tool_results_str += f"\nTool Call Result for `{msg.name}`:\n{msg.content}\n---"

    final_prompt = f"""
    Based on the original request, the student's report, and the results of the tool calls, please generate the final evaluation JSON.

    Original User Request:
    {state['user_prompt']}

    Student's Report:
    {state['output']}

    Tool Call Results:
    {tool_results_str if tool_results_str else "No tool calls were made."}

    Now, generate the complete, final JSON object based on all of this information.
    """
    
    logging.debug(f"Invoking final evaluation LLM with formatted prompt.")
    response = llm.invoke(final_prompt)
    logging.debug(f"Final evaluation LLM response:\n{response.content}")
    
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            eval_json = json.loads(json_match.group(0))
            state['evaluation_report'] = eval_json
            logging.debug("Successfully parsed final evaluation JSON.")
        else:
            raise json.JSONDecodeError("No JSON object found in the final LLM response.", response.content, 0)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse final evaluation JSON: {e}")
        state['evaluation_report'] = {"error": "Failed to generate valid final evaluation JSON."}
    return state

def should_call_tools(state: GraphState) -> str:
    logging.debug("Node: should_call_tools (router)")
    if not state.get('messages'):
        logging.error("Router error: No messages found to check for tool calls.")
        return "finalize_evaluation_json" # Fail gracefully

    last_message = state['messages'][-1]
    logging.debug(f"Inspecting last message for tool calls:\n{last_message}")
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logging.info("Decision: Tool call(s) present, routing to tool_node.")
        return "tool_node"
    else:
        logging.info("Decision: No tool calls, routing to finalize_evaluation_json.")
        return "finalize_evaluation_json"

def build_evaluation_graph(llm, prompts, tools):
    tool_node = ToolNode(tools)
    model_with_tools = llm.bind_tools(tools)
    graph = StateGraph(GraphState)
    graph.add_node("parse_claims", parse_claims)
    graph.add_node("generate_evaluation", lambda s: generate_evaluation(s, model_with_tools, prompts))
    graph.add_node("tool_node", tool_node)
    graph.add_node("finalize_evaluation_json", lambda s: finalize_evaluation_json(s, llm))
    graph.set_entry_point("parse_claims")
    graph.add_edge("parse_claims", "generate_evaluation")
    graph.add_conditional_edges("generate_evaluation", should_call_tools, {
        "tool_node": "tool_node",
        "finalize_evaluation_json": "finalize_evaluation_json"
    })
    graph.add_edge("tool_node", "finalize_evaluation_json")
    graph.add_edge("finalize_evaluation_json", END)
    return graph.compile()
