# supervisor_v8.py
# This file is fully reconciled with all technical specifications up to v8.0.
# It orchestrates the full Research -> Design -> Code pipeline via a Streamlit UI.
# This version includes the UnboundLocalError bugfix, auto-saving of code artifacts,
# and an intelligent file-reading tool to resolve path issues.

import os
import streamlit as st
import yaml
import json
import datetime
import tempfile
import logging
import glob

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_core.tools import tool

# --- Import all graph builders from the reconciled agent core ---
# Note: Ensure your local agent_core file is named 'agent_core_v8.py'
from agent_core_v8 import (
    GraphState, 
    build_planning_graph, 
    build_synthesis_graph, 
    build_evaluation_graph, 
    build_design_agent_graph, 
    build_coding_agent_graph
)

# --- Logger, Config, and Core App Functions ---
def setup_logger():
    """Initializes a session-wide logger."""
    logger_name = "ResearchAgentV8"
    # Define log_file outside the conditional block to prevent UnboundLocalError
    log_file = "agent_run_v8.log"
    if 'logger' not in st.session_state or st.session_state.logger.name != logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fh = logging.FileHandler(log_file, mode='w') 
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        st.session_state['logger'] = logger
        st.session_state['log_file'] = log_file
    return st.session_state['logger'], st.session_state['log_file']

logger, log_file = setup_logger()

# --- Constants and Environment Setup ---
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = st.secrets.get("GOOGLE_API_KEY", "")

ARTIFACT_DIR = "agent_outputs_v8"
PROMPTS_FILE = "prompts_v8.yaml"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DOC_LOADERS = {
    ".txt": TextLoader, ".md": TextLoader, ".py": TextLoader, 
    ".yaml": TextLoader, ".docx": UnstructuredWordDocumentLoader, ".pdf": PyPDFLoader
}

def init_session_state(): 
    """Initializes all necessary keys in the Streamlit session state."""
    keys_to_init = [
        'student_llm', 'teacher_llm', 'prompts', 'embeddings', 
        'retriever', 'raw_docs', 'current_state', 'run_phase'
    ]
    for key in keys_to_init:
        st.session_state.setdefault(key, None)

@st.cache_resource
def load_llms_and_embeddings():
    """Loads LLMs and embedding models, cached for the session."""
    student_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2, request_timeout=120)
    teacher_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1, request_timeout=120)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return student_llm, teacher_llm, embeddings

@st.cache_data
def load_prompts(): 
    """Loads prompts from the YAML file."""
    with open(PROMPTS_FILE, 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def process_uploaded_files(_files, _embeds):
    """Processes uploaded files into a FAISS vector store."""
    if not _files: return None, None
    
    docs, raw_docs = [], {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for f in _files:
            try:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                
                loader_cls = DOC_LOADERS.get(os.path.splitext(f.name)[1])
                if loader_cls:
                    loader = loader_cls(path)
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        d.metadata.update({'source': f.name})
                    docs.extend(loaded_docs)
                    raw_docs[f.name] = "\n".join([d.page_content for d in loaded_docs])
            except Exception as e:
                logger.error(f"Failed to load or process file {f.name}: {e}")
                st.error(f"Failed to process file: {f.name}")

    if not docs: return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, _embeds)
    return vectorstore.as_retriever(), raw_docs

# --- Agent Tools ---
@tool
def citation_retriever(doc_name: str) -> str:
    """Retrieves the full text content of a source document to verify claims."""
    logger.info(f"TOOL: Retrieving content for '{doc_name}'")
    return st.session_state.get('raw_docs', {}).get(doc_name, f"Error: Document '{doc_name}' not found in the knowledge base.")

@tool
def read_file(file_path: str) -> str:
    """
    Reads the content of a file. It intelligently searches for the file
    in the root directory (for source code) or in the artifact
    output directory (for generated artifacts like JSON).
    """
    logger.info(f"TOOL: Attempting to read file at smart path '{file_path}'")

    # First, try to read from the root directory (for .py, .yaml source files)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logger.info(f"Successfully read file from root path: {file_path}")
            return f.read()
    except FileNotFoundError:
        logger.warning(f"File '{file_path}' not found in root. Checking artifacts directory...")

    # If not in root, check the artifacts directory for the latest version
    try:
        base_name, extension = os.path.splitext(os.path.basename(file_path))
        # Search for files matching the pattern, e.g., "research_synthesis_*.json"
        search_pattern = os.path.join(ARTIFACT_DIR, f"{base_name}_*{extension}")
        
        list_of_files = glob.glob(search_pattern)
        if not list_of_files:
            raise FileNotFoundError

        # Find the most recent file
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Found latest artifact version: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            logger.info(f"Successfully read file from artifact path: {latest_file}")
            return f.read()
    
    except FileNotFoundError:
        logger.warning(f"File not found in artifacts either. Agent will treat '{file_path}' as a new file.")
        return f"Error: File not found at {file_path}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
        return f"Error reading file: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """(SIMULATED) Writes content to a file. In this UI, it only logs the action."""
    logger.info(f"TOOL: Simulating write to file at {file_path} with content length {len(content)}")
    return f"SIMULATED: File {file_path} would be written to successfully."

def save_artifacts(state: GraphState, run_phase: str):
    """Saves the generated artifacts to the output directory."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def save_content(content, filename):
        """Helper to save content to a file in the artifact directory."""
        if content is None:
            return
        try:
            path = os.path.join(ARTIFACT_DIR, filename)
            with open(path, "w", encoding="utf-8") as f:
                if isinstance(content, dict):
                    json.dump(content, f, indent=2)
                else:
                    f.write(content)
            st.success(f"Artifact saved: `{path}`")
        except Exception as e:
            st.error(f"Failed to save artifact {filename}: {e}")

    if run_phase == 'research_report':
        save_content(state.get('output'), f"research_report_{timestamp}.md")
        save_content(state.get('a2a_output'), f"research_synthesis_{timestamp}.json")
    
    elif run_phase == 'design_spec':
        save_content(state.get('design_spec'), f"design_spec_{timestamp}.md")
        save_content(state.get('design_synthesis_json'), f"design_synthesis_{timestamp}.json")
    
    elif run_phase == 'coding_execution':
        report = state.get('coding_execution_report', {})
        # Save the main JSON report for auditing
        save_content(report, f"coding_execution_report_{timestamp}.json")

        # Automatically save each proposed code file
        st.info("Saving proposed code files from execution report...")
        for artifact in report.get('final_code_artifacts', []):
            file_path = artifact.get('file_path')
            content = artifact.get('content')
            if file_path and content:
                # Create a safe, timestamped filename
                base_name, extension = os.path.splitext(os.path.basename(file_path))
                new_filename = f"{base_name}_{timestamp}{extension}"
                save_content(content, new_filename)

# --- UI Display Functions ---
def display_plan(plan):
    st.subheader("📝 Proposed Plan")
    if plan and "title" in plan and "plan_items" in plan:
        st.markdown(f"**Title:** {plan['title']}")
        for item in plan.get('plan_items', []):
            st.markdown(f"- **{item.get('section_id', '')} {item.get('title', '')}:** {item.get('description', '')}")
    else:
        st.error("Failed to generate a valid plan.")
        st.json(plan or {})

def display_evaluation(evaluation):
    st.subheader("Vetting Report")
    if not evaluation or "error" in evaluation:
        st.error("Failed to generate valid evaluation.")
        st.json(evaluation or {})
        return
        
    st.metric("Consistency Score", f"{evaluation.get('overall_consistency_score', 0.0):.2f}")
    st.info(f"**Notes:** {evaluation.get('consistency_notes', 'N/A')}")
    
    goal_status = evaluation.get('goal_alignment_check', 'FAIL')
    if goal_status == 'PASS':
        st.success(f"**Goal Alignment:** {goal_status} - {evaluation.get('goal_alignment_notes', '')}")
    else:
        st.error(f"**Goal Alignment:** {goal_status} - {evaluation.get('goal_alignment_notes', '')}")
        
    st.markdown("**Citation Audit:**")
    for audit in evaluation.get('citation_audit', []):
        status = audit.get('verification_status', 'FAIL')
        claim = audit.get('claim', 'N/A')
        tag = audit.get('citation_tag', 'N/A')
        if status == 'PASS':
            st.write(f"✅ **PASS:** {claim}")
        else:
            st.write(f"❌ **FAIL:** {claim} (Cited: {tag})")

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Agentic System v8", layout="wide")
    init_session_state()
    
    with st.sidebar:
        st.title("🔬 Agent Setup (v8)")
        if st.button("Start New Session", type="primary"):
            st.session_state.clear()
            st.rerun()
            
        uploaded_files = st.file_uploader(
            "1. Upload Knowledge Base", 
            type=list(DOC_LOADERS.keys()), 
            accept_multiple_files=True
        )
        
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing documents... This may take a moment."):
                st.session_state.retriever, st.session_state.raw_docs = process_uploaded_files(
                    tuple(uploaded_files), st.session_state.embeddings
                )
            if st.session_state.retriever:
                st.success("Knowledge base is ready.")
            else:
                st.error("Failed to build knowledge base. Please check files.")

        st.markdown("---")
        st.info(f"Logs: `{os.path.abspath(log_file)}`")
        st.info(f"Outputs: `{os.path.abspath(ARTIFACT_DIR)}`")

    st.title("Autonomous R&D Agent (v8)")
    
    # Lazy load LLMs and prompts
    if not st.session_state.get('student_llm'):
        st.session_state.student_llm, st.session_state.teacher_llm, st.session_state.embeddings = load_llms_and_embeddings()
        st.session_state.prompts = load_prompts()
    
    if not st.session_state.get('retriever'):
        st.info("Please upload documents and build the knowledge base to begin.")
        return

    current_phase = st.session_state.get('run_phase')

    # --- Full Pipeline State Machine ---
    if current_phase is None:
        st.header("Phase 1: Research")
        if st.button("📄 Generate Research Report", use_container_width=True):
            user_prompt = st.session_state.prompts['research_generation_task']
            # Initialize state for a fresh run
            st.session_state.current_state = GraphState(user_prompt=user_prompt, messages=[])
            with st.spinner("Agent is generating a research plan..."):
                plan_graph = build_planning_graph(st.session_state.student_llm, st.session_state.retriever, st.session_state.prompts)
                final_plan_state = plan_graph.invoke(st.session_state.current_state)
                st.session_state.current_state.update(final_plan_state)
            st.session_state.run_phase = "PLAN_APPROVAL"
            st.rerun()

    elif current_phase == "PLAN_APPROVAL":
        st.header("Phase 1: Plan Approval (HITL)")
        display_plan(st.session_state.current_state.get('plan'))
        if st.button("✅ Approve Plan & Begin Synthesis", use_container_width=True):
            st.session_state.run_phase = "SYNTHESIS"
            st.rerun()

    elif current_phase == "SYNTHESIS":
        st.header("Phase 2: Synthesis")
        with st.spinner("Agent is synthesizing the research report..."):
            st.session_state.current_state.update({"working_memory": {}, "completed_plan_items": []})
            synth_graph = build_synthesis_graph(st.session_state.student_llm, st.session_state.prompts)
            final_synth_state = synth_graph.invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_synth_state)
        st.session_state.run_phase = "EVALUATION"
        st.rerun()

    elif current_phase == "EVALUATION":
        st.header("Phase 3: Evaluation")
        st.markdown("**Synthesized Report:**")
        st.markdown(st.session_state.current_state.get('output', ''))
        with st.spinner("Agent is auditing the report..."):
            eval_graph = build_evaluation_graph(st.session_state.teacher_llm, st.session_state.prompts, [citation_retriever])
            final_eval_state = eval_graph.invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_eval_state)
        st.session_state.run_phase = "FINAL_REVIEW"
        st.rerun()

    elif current_phase == "FINAL_REVIEW":
        st.header("Phase 4: Research Review & Handoff to Design (HITL)")
        col1, col2 = st.columns(2)
        col1.markdown("**Final Report Draft:**")
        col1.markdown(st.session_state.current_state.get('output', ''))
        with col2:
            display_evaluation(st.session_state.current_state.get('evaluation_report'))
        
        with st.expander("View Agent-to-Agent (A2A) Research Synthesis Artifact"):
            st.json(st.session_state.current_state.get('a2a_output', {}))
        
        st.markdown("---")
        st.subheader("Feedback & Refinement")
        evaluation_report = st.session_state.current_state.get('evaluation_report', {})
        if evaluation_report.get('goal_alignment_check') != 'PASS':
             st.warning("Evaluation has raised issues. Please provide feedback to generate a new version.")

        feedback = st.text_area("If the report is not satisfactory, provide feedback here to re-run the process from the planning stage.")
        
        if st.button("Re-Plan with This Feedback", use_container_width=True):
            feedback_state = {
                "user_prompt": st.session_state.current_state['user_prompt'],
                "evaluation_report": st.session_state.current_state.get('evaluation_report'),
                "user_feedback": feedback,
                "messages": [],
            }
            st.session_state.current_state = GraphState(**feedback_state)
            st.session_state.run_phase = None # Reset to start planning again
            st.rerun()
        
        st.markdown("---")
        col_a, col_b = st.columns(2)
        if col_a.button("Accept & Save Research Artifacts", use_container_width=True):
            save_artifacts(st.session_state.current_state, 'research_report')

        if col_b.button("▶️ Generate Design Specification", use_container_width=True, type="primary"):
            st.session_state.run_phase = "DESIGN_GENERATION"
            st.rerun()

    elif current_phase == "DESIGN_GENERATION":
        st.header("Phase 5: Design Specification Generation")
        with st.spinner("Design Agent is generating the technical specification..."):
            design_graph = build_design_agent_graph(st.session_state.student_llm, st.session_state.prompts)
            final_design_state = design_graph.invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_design_state)
        st.session_state.run_phase = "DESIGN_REVIEW"
        st.rerun()

    elif current_phase == "DESIGN_REVIEW":
        st.header("Phase 6: Design Specification Review (HITL)")
        st.markdown(st.session_state.current_state.get('design_spec', "# No Design Spec Generated"))
        with st.expander("View Design Synthesis 'Work Order' (JSON)"):
            st.json(st.session_state.current_state.get('design_synthesis_json', {}))
        
        st.markdown("---")
        col_d, col_e = st.columns(2)
        if col_d.button("Accept & Save Design Artifacts", use_container_width=True):
            save_artifacts(st.session_state.current_state, 'design_spec')
        if col_e.button("▶️ Implement Code Changes", use_container_width=True, type="primary"):
            st.session_state.run_phase = "CODING"
            st.rerun()

    elif current_phase == "CODING":
        st.header("Phase 7: Coding Agent Execution")
        st.markdown("Starting Coding Agent with the following work order:")
        st.json(st.session_state.current_state.get('design_synthesis_json', {}))
        
        with st.spinner("Coding Agent is planning and executing changes..."):
            coding_tools = {"read_file": read_file, "write_file": write_file}
            coding_agent_graph = build_coding_agent_graph(st.session_state.student_llm, st.session_state.prompts, coding_tools)
            final_coding_state = coding_agent_graph.invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_coding_state)
            
        st.session_state.run_phase = "CODING_REVIEW"
        st.rerun()

    elif current_phase == "CODING_REVIEW":
        st.header("Phase 8: Coding Execution Review (HITL)")
        st.markdown("The Coding Agent has executed its plan. Below is the log and the final proposed artifacts.")
        
        st.subheader("Execution Log")
        st.code('\n'.join(st.session_state.current_state.get('coding_log', ["No log generated."])), language='bash')
        
        st.subheader("Final Execution Report")
        st.json(st.session_state.current_state.get('coding_execution_report', {}))
        
        st.markdown("---")
        if st.button("Accept & Save All Final Artifacts", use_container_width=True):
            save_artifacts(st.session_state.current_state, 'coding_execution')
        
        if st.button("Finish and Restart Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()

