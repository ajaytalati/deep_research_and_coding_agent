# agent_core_v2.py
# v2.2: Refactored evaluation graph into a single, robust tool-enabled call.

import json
import re
import logging
from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- FIX: Get the same named logger as the supervisor ---
logger = logging.getLogger("ResearchAgentV2")

class GraphState(TypedDict):
    user_prompt: str
    documents: List[Dict]
    plan: Optional[Dict]
    output: str
    evaluation_report: Optional[Dict]
    user_feedback: Optional[str]
    messages: List[BaseMessage]
    working_memory: Optional[Dict]
    completed_plan_items: List[str]
    parsed_claims: List[Dict]

# --- Planning and Synthesis Graphs (Now with full code) ---
def retrieve_context(state: GraphState, retriever):
    logger.debug("Node: retrieve_context")
    docs = retriever.get_relevant_documents(state["user_prompt"])
    state['documents'] = [{"name": doc.metadata.get('source', 'unknown'), "content": doc.page_content} for doc in docs]
    logger.debug(f"Retrieved {len(state['documents'])} documents.")
    return state

def generate_plan(state: GraphState, llm, prompts):
    logger.debug("Node: generate_plan")
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    prompt = prompts['create_plan_prompt'].format(context=context_str, user_prompt=state['user_prompt'])
    
    response_content = llm.invoke(prompt).content
    logger.debug(f"LLM Raw Plan Response:\n{response_content}")
    
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
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", lambda s: retrieve_context(s, retriever))
    graph.add_node("generate_plan", lambda s: generate_plan(s, llm, prompts))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate_plan")
    graph.add_edge("generate_plan", END)
    return graph.compile()

def execute_synthesis_step(state: GraphState, llm, prompts):
    plan = state.get('plan', {})
    completed = state.get('completed_plan_items', [])
    current_item = next((item for item in plan.get('plan_items', []) if item['section_id'] not in completed), None)
    
    if not current_item:
        return state

    logger.debug(f"Node: execute_synthesis_step for section {current_item['section_id']}")
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    plan_str = json.dumps(plan, indent=2)
    prompt = prompts['synthesis_step_prompt'].format(
        context=context_str, plan_str=plan_str, 
        item_title=current_item['title'], item_description=current_item['description']
    )
    generated_content = llm.invoke(prompt).content
    state['working_memory'][current_item['section_id']] = generated_content
    state['completed_plan_items'].append(current_item['section_id'])
    return state

def assemble_draft(state: GraphState):
    logger.debug("Node: assemble_draft")
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
        plan = state.get('plan', {})
        completed_count = len(state.get('completed_plan_items', []))
        total_count = len(plan.get('plan_items', []))
        if completed_count >= total_count:
            return "assemble_draft"
        else:
            return "execute_synthesis_step"

    graph.set_entry_point("execute_synthesis_step")
    graph.add_conditional_edges("execute_synthesis_step", synthesis_router)
    graph.add_edge("assemble_draft", END)
    return graph.compile()

# --- NEW, SIMPLIFIED & ROBUST EVALUATION GRAPH ---

def parse_claims_node(state: GraphState):
    logger.debug("--- Entering Evaluation Graph ---")
    logger.debug(f"Node: parse_claims_node on output (first 300 chars):\n{state.get('output', '')[:300]}...")
    text = state.get('output', '')
    # This regex is simple; more complex claims might be missed but it's a good start.
    claim_pattern = re.compile(r'([^\.!?]*?\[Source:[^\]]+\][\s\.]*)')
    claims = claim_pattern.findall(text)
    state['parsed_claims'] = [{"claim": claim.strip()} for claim in claims]
    logger.debug(f"Found {len(state['parsed_claims'])} potential claims to audit.")
    return state

# --- FIX: This is the new, single node for evaluation ---
def generate_evaluation_node(state: GraphState, llm, prompts):
    logger.debug("Node: generate_evaluation_node")
    
    # The prompt now instructs the LLM to do everything in one go.
    prompt = prompts['evaluation_prompt_v2'].format(user_prompt=state['user_prompt'], output=state['output'])
    
    logger.debug(f"Invoking evaluation LLM with tool-binding and direct-to-JSON instruction.")
    
    # The LLM will see the prompt, see the tools, use them internally,
    # and then generate the final JSON directly.
    response = llm.invoke(prompt)
    
    logger.debug(f"Raw response from evaluation LLM:\n{response.content}")
    
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
    # The tool-enabled model is passed to the single generation node.
    model_with_tools = llm.bind_tools(tools)
    
    graph = StateGraph(GraphState)
    
    # The graph is now much simpler.
    graph.add_node("parse_claims", parse_claims_node)
    graph.add_node("generate_evaluation", lambda s: generate_evaluation_node(s, model_with_tools, prompts))

    graph.set_entry_point("parse_claims")
    graph.add_edge("parse_claims", "generate_evaluation")
    graph.add_edge("generate_evaluation", END)
    
    return graph.compile()

