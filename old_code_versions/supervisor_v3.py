# supervisor_v3.py
# Implements TDS v5.0: Saves the new A2A JSON artifact.
# Updated to v3 to reflect major architectural changes.

import os
import streamlit as st
import yaml
import json
import datetime
import tempfile
import logging
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_core.tools import tool

# --- v3: Import from the new agent core ---
from agent_core_v3 import GraphState, build_planning_graph, build_synthesis_graph, build_evaluation_graph

# --- Robust, Session-Based Logger Setup ---
def setup_logger():
    """
    Sets up a dedicated logger for the agent application.
    Ensures the logger is configured only once per Streamlit session.
    """
    logger_name = "ResearchAgentV3"
    if 'logger' not in st.session_state or st.session_state.logger.name != logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            log_file = "agent_run_v3.log"
            fh = logging.FileHandler(log_file, mode='w') 
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        st.session_state['logger'] = logger
        st.session_state['log_file'] = log_file
        
    return st.session_state['logger'], st.session_state['log_file']

logger, log_file = setup_logger()

# --- Global Configuration ---
if 'GOOGLE_API_KEY' not in os.environ: os.environ['GOOGLE_API_KEY'] = st.secrets.get("GOOGLE_API_KEY", "")

# --- v3: Update constants for new file names ---
ARTIFACT_DIR, PROMPTS_FILE = "agent_outputs_v3", "prompts_v3.yaml"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
DOC_LOADERS = {".txt": TextLoader, ".md": TextLoader, ".py": TextLoader, ".docx": UnstructuredWordDocumentLoader, ".pdf": PyPDFLoader}

def init_session_state():
    """Initializes all necessary keys in the Streamlit session state."""
    logger.info("Initializing session state for v3.")
    keys_to_init = ['student_llm', 'teacher_llm', 'prompts', 'embeddings', 'retriever', 'raw_docs', 'current_state', 'run_phase', 'messages']
    for key in keys_to_init:
        if key not in st.session_state:
            st.session_state[key] = None

@st.cache_resource
def load_llms_and_embeddings():
    """Loads LLMs and embedding models, cached for the session."""
    logger.info("Loading LLMs and Embeddings...")
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🚨 GOOGLE_API_KEY not set!"); logger.error("FATAL: GOOGLE_API_KEY not set!"); st.stop()
    student_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
    teacher_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    logger.info("LLMs and Embeddings loaded successfully.")
    return student_llm, teacher_llm, embeddings

@st.cache_data
def load_prompts():
    """Loads prompts from the YAML file, cached for the session."""
    logger.info("Loading prompts from %s", PROMPTS_FILE)
    with open(PROMPTS_FILE, 'r') as f: prompts = yaml.safe_load(f)
    logger.info("Prompts loaded successfully.")
    return prompts

@st.cache_resource
def process_uploaded_files(_files, _embeds):
    """Processes uploaded files into a FAISS vector store retriever."""
    if not _files: return None, None
    logger.info("Processing %d uploaded files...", len(_files))
    docs, raw_docs = [], {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for f in _files:
            path = os.path.join(temp_dir, f.name)
            with open(path, "wb") as out: out.write(f.getbuffer())
            if loader := DOC_LOADERS.get(os.path.splitext(f.name)[1]):
                try:
                    loaded_docs = loader(path).load()
                    for doc in loaded_docs: doc.metadata['source'] = f.name
                    docs.extend(loaded_docs)
                    raw_docs[f.name] = "\n".join([d.page_content for d in loaded_docs])
                except Exception as e:
                    logger.error("Failed to load file %s: %s", f.name, e)
    if not docs:
        logger.warning("No documents could be loaded from uploaded files.")
        return None, None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    retriever = FAISS.from_documents(text_splitter.split_documents(docs), _embeds).as_retriever()
    logger.info("Knowledge base created successfully with %d documents.", len(raw_docs))
    return retriever, raw_docs

@tool
def citation_retriever(doc_name: str) -> str:
    """
    Retrieves the full text content of a specific source document from the knowledge base.
    This tool is used by the evaluation agent to verify claims.
    """
    logger.info("TOOL CALL: citation_retriever(doc_name='%s')", doc_name)
    raw_docs = st.session_state.get('raw_docs', {})
    content = raw_docs.get(doc_name, f"Error: Document '{doc_name}' not found.")
    if "Error" in content: logger.warning("citation_retriever failed to find document: %s", doc_name)
    return content

def save_artifacts(state: GraphState):
    """
    Saves the final generated artifacts (both .md and .json) to the output directory.
    """
    logger.debug("Attempting to save artifacts...")
    run_phase = state.get('run_phase', 'unknown')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the human-readable Markdown report
    output_content_md = state.get('output')
    if output_content_md and isinstance(output_content_md, str) and output_content_md.strip():
        try:
            filename_md = f"agent_output_{run_phase}_{timestamp}.md"
            path_md = os.path.join(ARTIFACT_DIR, filename_md)
            with open(path_md, "w", encoding="utf-8") as f: f.write(output_content_md)
            st.success(f"Report saved to `{path_md}`")
            logger.info("Successfully saved report to %s", path_md)
        except Exception as e:
            st.error(f"Failed to save Markdown artifact: {e}"); logger.error("Failed to save MD artifact: %s", e, exc_info=True)
    else:
        logger.warning("Markdown output was empty or invalid. Nothing saved.")

    # --- v3 MODIFICATION (TDS v5.0): Save the A2A JSON artifact ---
    output_content_json = state.get('a2a_output')
    if output_content_json and isinstance(output_content_json, dict) and "error" not in output_content_json:
        try:
            filename_json = f"agent_output_{run_phase}_{timestamp}.json"
            path_json = os.path.join(ARTIFACT_DIR, filename_json)
            with open(path_json, "w", encoding="utf-8") as f: json.dump(output_content_json, f, indent=2)
            st.success(f"A2A artifact saved to `{path_json}`")
            logger.info("Successfully saved A2A artifact to %s", path_json)
        except Exception as e:
            st.error(f"Failed to save JSON artifact: {e}"); logger.error("Failed to save JSON artifact: %s", e, exc_info=True)
    else:
        logger.warning("A2A JSON output was empty or invalid. Nothing saved.")

# --- UI Display Functions ---

def display_plan(plan):
    """Renders the generated plan in the UI."""
    st.subheader("📝 Proposed Plan")
    if plan and "title" in plan and "plan_items" in plan:
        st.markdown(f"**Title:** {plan['title']}")
        for item in plan.get('plan_items', []): st.markdown(f"- **{item['section_id']} {item['title']}:** {item['description']}")
    else:
        st.error("Failed to generate a valid plan."); st.json(plan or {"error": "Plan object is None or invalid."})

def display_evaluation(evaluation):
    """Renders the evaluation report in the UI."""
    st.subheader("Vetting Report")
    if not evaluation or "error" in evaluation:
        st.error("Failed to generate a valid evaluation."); st.json(evaluation or {"error": "Evaluation object is None or invalid."}); return
    st.metric("Consistency Score", f"{evaluation.get('overall_consistency_score', 0.0):.2f}")
    st.info(f"**Notes:** {evaluation.get('consistency_notes', 'N/A')}")
    goal_status = evaluation.get('goal_alignment_check', 'FAIL')
    if goal_status == 'PASS': st.success(f"**Goal Alignment:** {goal_status} - {evaluation.get('goal_alignment_notes', '')}")
    else: st.error(f"**Goal Alignment:** {goal_status} - {evaluation.get('goal_alignment_notes', '')}")
    st.markdown("**Citation Audit:**")
    audit_results = evaluation.get('citation_audit', [])
    if not audit_results: st.warning("No citations were audited.")
    for audit in audit_results:
        status = audit.get('verification_status', 'FAIL'); claim = audit.get('claim', 'N/A'); tag = audit.get('citation_tag', 'N/A')
        if status == 'PASS': st.write(f"✅ **PASS:** {claim}")
        else: st.write(f"❌ **FAIL:** {claim} (Cited: {tag})")

# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Agentic System v3", layout="wide")
    init_session_state()
    
    with st.sidebar:
        st.title("🔬 Agent Setup (v3)")
        if st.button("Start New Session", type="primary"):
            # Clear log file and session state
            if 'log_file' in st.session_state and os.path.exists(st.session_state['log_file']):
                with open(st.session_state['log_file'], "w"): pass
            for key in list(st.session_state.keys()):
                # Keep the logger configuration
                if 'logger' not in key and 'log_file' not in key:
                    del st.session_state[key]
            st.rerun()

        uploaded_files = st.file_uploader("1. Upload Knowledge Base", type=list(DOC_LOADERS.keys()), accept_multiple_files=True)
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing documents..."):
                st.session_state.retriever, st.session_state.raw_docs = process_uploaded_files(tuple(uploaded_files), st.session_state.embeddings)
            if st.session_state.retriever: st.success("Knowledge base ready.")
            else: st.warning("No files processed or failed to build retriever.")
        st.markdown("---")
        st.info(f"Logs are in: `{os.path.abspath(log_file)}`")
        st.info(f"Outputs are in: `{os.path.abspath(ARTIFACT_DIR)}`")

    st.title("Professional Grade Research Agent (v3.0)")
    
    # Load models and prompts if they aren't in the session state
    if 'student_llm' not in st.session_state or st.session_state.student_llm is None:
        st.session_state.student_llm, st.session_state.teacher_llm, st.session_state.embeddings = load_llms_and_embeddings()
        st.session_state.prompts = load_prompts()

    if not st.session_state.retriever:
        st.info("Please upload documents and build a knowledge base to begin.")
        return

    current_phase = st.session_state.get('run_phase')

    # --- PHASE 1: PLANNING ---
    if current_phase is None:
        st.header("Phase 1: Planning")
        if st.button("📄 Generate Research Report", use_container_width=True):
            st.session_state.run_request = st.session_state.prompts['research_generation_task']
        
        if st.session_state.get("run_request"):
            user_prompt = st.session_state.pop("run_request")
            st.session_state.current_state = GraphState(user_prompt=user_prompt, messages=[])
            logger.info("Starting PLANNING phase for prompt: %s", user_prompt[:100])
            with st.spinner("Student agent is generating a plan..."):
                planning_graph = build_planning_graph(st.session_state.student_llm, st.session_state.retriever, st.session_state.prompts)
                final_plan_state = planning_graph.invoke(st.session_state.current_state)
                st.session_state.current_state.update(final_plan_state)
            st.session_state.run_phase = "PLAN_APPROVAL"
            st.rerun()
    
    # --- PHASE 2: PLAN APPROVAL (HITL) ---
    elif current_phase == "PLAN_APPROVAL":
        st.header("Phase 1: Plan Approval (HITL)")
        display_plan(st.session_state.current_state.get('plan'))
        if st.button("✅ Approve Plan & Begin Synthesis", use_container_width=True):
            st.session_state.run_phase = "SYNTHESIS"; logger.info("Plan approved. Moving to SYNTHESIS."); st.rerun()
    
    # --- PHASE 3: SYNTHESIS ---
    elif current_phase == "SYNTHESIS":
        st.header("Phase 2: Synthesis")
        with st.spinner("Student agent is generating the report..."):
            logger.info("Starting SYNTHESIS phase.")
            synthesis_graph = build_synthesis_graph(st.session_state.student_llm, st.session_state.prompts)
            st.session_state.current_state.update({"working_memory": {}, "completed_plan_items": []})
            final_synth_state = synthesis_graph.invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_synth_state)
            logger.info("SYNTHESIS phase complete.")
        st.session_state.run_phase = "EVALUATION"; st.rerun()

    # --- PHASE 4: EVALUATION ---
    elif current_phase == "EVALUATION":
        st.header("Phase 3: Evaluation")
        st.markdown("**Synthesized Report:**"); st.markdown(st.session_state.current_state.get('output', ''))
        with st.spinner("Teacher agent is auditing the report..."):
            logger.info("Starting EVALUATION phase.")
            evaluation_graph = build_evaluation_graph(st.session_state.teacher_llm, st.session_state.prompts, [citation_retriever])
            final_eval_state = evaluation_graph.invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_eval_state)
            logger.info("EVALUATION phase complete.")
        st.session_state.run_phase = "FINAL_REVIEW"; st.rerun()

    # --- PHASE 5: FINAL REVIEW & REFINEMENT (HITL) ---
    elif current_phase == "FINAL_REVIEW":
        st.header("Phase 4: Final Review & Refinement (HITL)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Final Report Draft:**"); st.markdown(st.session_state.current_state.get('output', ''))
        with col2:
            display_evaluation(st.session_state.current_state.get('evaluation_report'))
        
        # Display the generated A2A JSON artifact for debugging/verification
        with st.expander("View Agent-to-Agent (A2A) JSON Artifact"):
            st.json(st.session_state.current_state.get('a2a_output', {"info": "Not generated or an error occurred."}))

        st.markdown("---")
        if st.button("Accept Final Report & Save Artifacts", use_container_width=True):
            st.session_state.current_state['run_phase'] = "research_report"
            save_artifacts(st.session_state.current_state)
            st.success("Process complete! Artifacts saved.")
            logger.info("Process complete. Final artifacts accepted by user.")
            st.session_state.run_phase = None
            st.session_state.current_state = None
            st.balloons()
        
        with st.expander("Or, provide feedback and restart"):
            feedback = st.text_area("Your feedback to guide the next planning cycle:")
            if st.button("Re-Plan with This Feedback"):
                logger.info("User requested restart with feedback: %s", feedback)
                original_prompt = st.session_state.current_state['user_prompt']
                # Reset state for the new loop
                st.session_state.current_state = GraphState(
                    user_prompt=original_prompt,
                    user_feedback=feedback,
                    evaluation_report=st.session_state.current_state.get('evaluation_report'),
                    messages=[]
                )
                st.session_state.run_phase = None
                st.rerun()

if __name__ == "__main__":
    main()

