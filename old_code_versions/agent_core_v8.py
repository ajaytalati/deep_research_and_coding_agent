# agent_core_v8.py
# This file is fully reconciled with all technical specifications up to v8.0.
# It implements the Research, Design, and Coding agent graphs.

import json
import re
import logging
import datetime
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# --- Logger Setup ---
# Ensures that the logger is configured once and used across the application.
logger = logging.getLogger("ResearchAgentV8")

# --- Central State Definition ---
class GraphState(TypedDict):
    """
    Defines the central state that is passed between all nodes in all agent graphs.
    This TypedDict acts as the shared memory for the entire multi-agent system.
    """
    # Core inputs
    user_prompt: str
    user_feedback: Optional[str]

    # Research Agent state
    documents: List[Dict]
    plan: Optional[Dict]
    output: str
    evaluation_report: Optional[Dict]
    working_memory: Optional[Dict]
    completed_plan_items: List[str]
    a2a_output: Optional[Dict] # Agent-to-Agent research synthesis

    # Design Agent state
    design_spec: Optional[str]
    design_synthesis_json: Optional[Dict] # Agent-to-Agent design work order

    # Coding Agent state
    coding_plan: Optional[Dict]
    coding_log: List[str]
    in_memory_files: Dict[str, str] # In-memory file cache for stateful execution
    coding_execution_report: Optional[Dict]

    # LangChain specific
    messages: List[BaseMessage]

def parse_json_from_response(content: str) -> Optional[Dict]:
    """
    Safely extracts and parses a JSON object from a model's string response.
    Handles responses that may include markdown code blocks or other text.
    """
    # Look for a JSON object within ```json ... ```
    match = re.search(r'```json\n(\{.*?\})\n```', content, re.DOTALL)
    if not match:
        # If not found, look for any JSON object in the string
        match = re.search(r'(\{.*?\})', content, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nContent was: {content}")
            return {"error": f"JSON Parsing Failed: {e}", "content": content}
    logger.error(f"No JSON object found in response.\nContent was: {content}")
    return {"error": "No JSON object found", "content": content}

# --- RESEARCH AGENT: PLANNING GRAPH (TDS v1.0, v4.0) ---
def retrieve_context(state: GraphState, retriever):
    """Node: Retrieves relevant documents from the vector store."""
    logger.debug("Node: retrieve_context")
    docs = retriever.invoke(state["user_prompt"])
    state['documents'] = [{"name": doc.metadata.get('source', 'unknown'), "content": doc.page_content} for doc in docs]
    return state

def generate_plan(state: GraphState, llm, prompts):
    """
    Node: Generates a structured plan (table of contents).
    MODIFIED to include the user feedback loop from TDS v1.0.
    """
    logger.debug("Node: generate_plan")
    
    # --- RESTORED LOGIC (Consolidated TDS v1.0) ---
    # This logic handles the iterative refinement loop.
    feedback_prefix = ""
    if state.get('user_feedback') or (state.get('evaluation_report') and state['evaluation_report'].get('goal_alignment_check') != 'PASS'):
        logger.info("Revising plan based on feedback.")
        eval_report_str = json.dumps(state.get('evaluation_report'), indent=2)
        user_feedback_str = state.get('user_feedback', "No additional user feedback provided.")
        feedback_prefix = (
            "You are in a revision cycle. A previous attempt was evaluated and requires changes. "
            "You MUST create a new plan that addresses the following critiques.\n\n"
            f"--- CRITIQUE FROM EVALUATION REPORT ---\n{eval_report_str}\n\n"
            f"--- ADDITIONAL USER FEEDBACK ---\n{user_feedback_str}\n\n"
            "--- REVISED PLAN ---\n"
        )
    # --- END OF RESTORED LOGIC ---

    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    prompt = prompts['create_plan_prompt'].format(context=context_str, user_prompt=state['user_prompt'])
    
    final_prompt = feedback_prefix + prompt
    
    response_content = llm.invoke(final_prompt).content
    state['plan'] = parse_json_from_response(response_content)
    return state

def build_planning_graph(llm, retriever, prompts):
    """Factory for the planning graph."""
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", lambda s: retrieve_context(s, retriever))
    graph.add_node("generate_plan", lambda s: generate_plan(s, llm, prompts))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate_plan")
    graph.add_edge("generate_plan", END)
    return graph.compile()

# --- RESEARCH AGENT: SYNTHESIS GRAPH (TDS v1.0, v5.0) ---
def execute_synthesis_step(state: GraphState, llm, prompts):
    """Node: Generates content for one section of the plan at a time."""
    plan, completed = state.get('plan', {}), state.get('completed_plan_items', [])
    current_item = next((item for item in plan.get('plan_items', []) if item['section_id'] not in completed), None)
    
    if not current_item:
        logger.debug("All synthesis steps are complete.")
        return state
        
    logger.debug(f"Executing synthesis for section: {current_item['section_id']}")
    context_str = "\n---\n".join([f"Source: {doc['name']}\n\n{doc['content']}" for doc in state['documents']])
    plan_str = json.dumps(plan, indent=2)
    prompt = prompts['synthesis_step_prompt'].format(context=context_str, plan_str=plan_str, item_title=current_item['title'], item_description=current_item['description'])
    
    state['working_memory'][current_item['section_id']] = llm.invoke(prompt).content
    state['completed_plan_items'].append(current_item['section_id'])
    return state

def assemble_and_finalize_draft(state: GraphState, llm, prompts):
    """
    Node: Assembles the final report and generates the A2A synthesis artifact.
    MODIFIED to include the A2A artifact generation from TDS v5.0.
    """
    logger.debug("Node: assemble_and_finalize_draft")
    plan_items = state.get('plan', {}).get('plan_items', [])
    ordered_sections = sorted(plan_items, key=lambda x: str(x.get('section_id', '')))
    full_draft = "\n\n---\n\n".join([f"## {item['title']}\n\n{state['working_memory'].get(item['section_id'], '')}" for item in ordered_sections])
    
    # Programmatically add numeric indices to citations for traceability
    citation_counter = 0
    def replace_with_index(match):
        nonlocal citation_counter
        citation_counter += 1
        return f"{match.group(1)}[{citation_counter}]"
    state['output'] = re.sub(r'(\[Source: [^\]]+\])', replace_with_index, full_draft)

    # --- NEW LOGIC (TDS v5.0) ---
    logger.info("Generating A2A Research Synthesis artifact...")
    a2a_prompt = prompts['create_a2a_synthesis_prompt'].format(
        final_report_text=state['output'],
        current_date=datetime.date.today().isoformat()
    )
    response = llm.invoke(a2a_prompt)
    state['a2a_output'] = parse_json_from_response(response.content)
    if 'error' in state['a2a_output']:
        logger.error("Failed to generate A2A research synthesis JSON.")
    else:
        logger.info("Successfully generated A2A research synthesis artifact.")
    # --- END OF NEW LOGIC ---

    return state

def build_synthesis_graph(llm, prompts):
    """Factory for the synthesis graph."""
    graph = StateGraph(GraphState)
    graph.add_node("execute_synthesis_step", lambda s: execute_synthesis_step(s, llm, prompts))
    graph.add_node("assemble_and_finalize_draft", lambda s: assemble_and_finalize_draft(s, llm, prompts))
    
    def synthesis_router(state):
        plan, completed = state.get('plan', {}), state.get('completed_plan_items', [])
        if len(completed) >= len(plan.get('plan_items', [])):
            return "assemble_and_finalize_draft"
        return "execute_synthesis_step"
        
    graph.set_entry_point("execute_synthesis_step")
    graph.add_conditional_edges("execute_synthesis_step", synthesis_router)
    graph.add_edge("assemble_and_finalize_draft", END)
    return graph.compile()

# --- RESEARCH AGENT: EVALUATION GRAPH (TDS v4.0) ---
def generate_evaluation_node(state: GraphState, llm, prompts):
    """Node: Performs a single-call, tool-augmented evaluation of the report."""
    logger.debug("Node: generate_evaluation")
    prompt = prompts['evaluation_prompt_v2'].format(user_prompt=state['user_prompt'], output=state['output'])
    response = llm.invoke(prompt)
    state['evaluation_report'] = parse_json_from_response(response.content)
    return state

def build_evaluation_graph(llm, prompts, tools):
    """Factory for the evaluation graph."""
    # Bind the tools to the LLM for this specific graph
    model_with_tools = llm.bind_tools(tools)
    graph = StateGraph(GraphState)
    graph.add_node("generate_evaluation", lambda s: generate_evaluation_node(s, model_with_tools, prompts))
    graph.set_entry_point("generate_evaluation")
    graph.add_edge("generate_evaluation", END)
    return graph.compile()

# --- NEW DESIGN AGENT GRAPH (TDS v5.1 & v6.0) ---
def generate_design_spec_node(state: GraphState, llm, prompts):
    """Node: Generates both the markdown design spec and the JSON work order."""
    logger.debug("--- Entering Design Agent Graph ---")
    research_synthesis = state.get('a2a_output')
    if not research_synthesis or 'error' in research_synthesis:
        state['design_spec'] = "# ERROR: Research Synthesis artifact is missing or invalid."
        state['design_synthesis_json'] = {"error": "Upstream research synthesis artifact missing."}
        return state

    # 1. Generate the human-readable Markdown Design Spec
    logger.info("Generating Markdown Technical Design Specification...")
    md_prompt = prompts['create_design_spec_prompt'].format(research_synthesis_json=json.dumps(research_synthesis, indent=2))
    state['design_spec'] = llm.invoke(md_prompt).content
    logger.info("Successfully generated markdown design specification.")

    # 2. Generate the machine-readable JSON Design Synthesis (Work Order)
    logger.info("Generating JSON Design Synthesis artifact...")
    json_prompt = prompts['create_design_synthesis_prompt'].format(
        design_spec_md=state['design_spec'],
        current_date=datetime.date.today().isoformat()
    )
    response = llm.invoke(json_prompt)
    state['design_synthesis_json'] = parse_json_from_response(response.content)
    if 'error' in state['design_synthesis_json']:
        logger.error("Failed to generate design synthesis JSON from the design spec.")
    else:
        logger.info("Successfully generated design synthesis JSON artifact.")

    return state

def build_design_agent_graph(llm, prompts):
    """Factory function for the Design Agent graph."""
    graph = StateGraph(GraphState)
    graph.add_node("generate_design_spec", lambda s: generate_design_spec_node(s, llm, prompts))
    graph.set_entry_point("generate_design_spec")
    graph.add_edge("generate_design_spec", END)
    return graph.compile()
# --- END OF NEW DESIGN AGENT ---

# --- CODING AGENT GRAPH (TDS v7.0 & v8.0) ---
def coding_planning_node(state: GraphState, llm, prompts) -> GraphState:
    """Node: Creates a high-level plan of which files to modify."""
    logger.debug("--- Entering Coding Agent Graph: Planning ---")
    work_order = state.get('design_synthesis_json')
    if not work_order or 'error' in work_order:
        state['coding_plan'] = {"error": "Upstream design artifact missing or invalid."}
        state['coding_log'] = ["Execution skipped due to missing/invalid design work order."]
        return state

    prompt = prompts['create_coding_plan_prompt'].format(design_synthesis_json=json.dumps(work_order, indent=2))
    response_content = llm.invoke(prompt).content
    state['coding_plan'] = parse_json_from_response(response_content)
    state['coding_log'] = [] # Initialize log
    state['in_memory_files'] = {} # Initialize in-memory file cache
    return state

def code_execution_node(state: GraphState, llm, prompts, tools: Dict[str, callable]) -> GraphState:
    """Node: Executes the coding plan using a stateful Read-Think-Write loop."""
    logger.debug("--- Entering Coding Agent Graph: Execution ---")
    plan = state.get('coding_plan', {})
    log = state.get('coding_log', [])
    in_memory_files = state.get('in_memory_files', {})

    if "error" in plan:
        log.append("Execution skipped due to planning error.")
        state['coding_log'] = log
        return state

    for step in plan.get('steps', []):
        description, file_path = step.get('description'), step.get('file_path')
        if not description or not file_path:
            log.append(f"WARNING: Skipping invalid plan step: {step}")
            continue

        log.append(f"--- Executing Step: {description} ---")
        
        # 1. READ (from cache first, then from disk tool)
        log.append(f"READ: Checking in-memory cache for '{file_path}'")
        if file_path not in in_memory_files:
            log.append(f"Cache miss. Reading from disk: '{file_path}'")
            read_tool = tools['read_file']
            in_memory_files[file_path] = read_tool.invoke({"file_path": file_path})
        
        original_content = in_memory_files[file_path]
        log.append(f"READ_RESULT: (Using content of length {len(original_content)} chars)")
        if "Error:" in original_content and "File not found" not in original_content:
            log.append(f"ERROR: Halting step due to read failure. {original_content}")
            continue

        # 2. THINK (Generate new content based on current state)
        log.append(f"THINK: Generating new content for '{file_path}'...")
        prompt = prompts['generate_file_content_prompt'].format(
            instruction=description,
            original_content=original_content if "File not found" not in original_content else "# This is a new file."
        )
        new_content = llm.invoke(prompt).content
        # Clean up the response, removing markdown code fences
        new_content = re.sub(r'^```(python|yaml|json)?\n|```$', '', new_content, flags=re.MULTILINE).strip()
        log.append(f"THINK_RESULT: New content generated (length: {len(new_content)}).")
        
        # 3. UPDATE IN-MEMORY STATE (instead of writing to disk)
        log.append(f"UPDATE: Updating in-memory cache for '{file_path}'")
        in_memory_files[file_path] = new_content

    state['coding_log'] = log
    state['in_memory_files'] = in_memory_files
    return state

def finalize_coding_report_node(state: GraphState, llm, prompts) -> GraphState:
    """Node: Generates the final JSON execution report from the in-memory files."""
    logger.debug("--- Entering Coding Agent Graph: Finalization ---")
    in_memory_files = state.get('in_memory_files', {})
    design_spec = state.get('design_synthesis_json', {})

    if not in_memory_files:
        if not state.get('coding_log'): # If there was no planning error, this means no files were planned for modification
             state['coding_log'].append("No file modifications were planned or executed.")
        state['coding_execution_report'] = {"error": "No files were modified in memory."}
        return state

    prompt = prompts['create_coding_report_prompt'].format(
        final_file_contents_json=json.dumps(in_memory_files, indent=2),
        current_date=datetime.date.today().isoformat(),
        design_spec_version=design_spec.get("metadata", {}).get("design_spec_version", "N/A")
    )
    response_content = llm.invoke(prompt).content
    state['coding_execution_report'] = parse_json_from_response(response_content)
    logger.info("Successfully generated coding execution report.")
    return state

def build_coding_agent_graph(llm, prompts, tools):
    """Factory function for the v8 stateful Coding Agent."""
    graph = StateGraph(GraphState)
    graph.add_node("generate_coding_plan", lambda s: coding_planning_node(s, llm, prompts))
    graph.add_node("execute_code_read_think_write", lambda s: code_execution_node(s, llm, prompts, tools))
    graph.add_node("finalize_coding_report", lambda s: finalize_coding_report_node(s, llm, prompts))
    
    graph.set_entry_point("generate_coding_plan")
    graph.add_edge("generate_coding_plan", "execute_code_read_think_write")
    graph.add_edge("execute_code_read_think_write", "finalize_coding_report")
    graph.add_edge("finalize_coding_report", END)
    
    return graph.compile()

