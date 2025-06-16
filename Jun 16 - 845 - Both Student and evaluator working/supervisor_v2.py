# supervisor_v2.py
# v2.3: Definitive fix for NameError and logging issues.

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
from agent_core_v2 import GraphState, build_planning_graph, build_synthesis_graph, build_evaluation_graph

# --- Robust, Session-Based Logger Setup ---
def setup_logger():
    """
    Sets up a dedicated logger for the agent application.
    Ensures the logger is configured only once per Streamlit session.
    """
    if 'logger' not in st.session_state:
        logger = logging.getLogger("ResearchAgentV2")
        logger.setLevel(logging.DEBUG)
        
        # Prevent adding handlers if they already exist
        if not logger.handlers:
            log_file = "agent_run_v2.log"
            # Use 'w' mode to clear the log file for each new session start
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

ARTIFACT_DIR, PROMPTS_FILE = "agent_outputs", "prompts.yaml"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
DOC_LOADERS = {".txt": TextLoader, ".md": TextLoader, ".py": TextLoader, ".docx": UnstructuredWordDocumentLoader, ".pdf": PyPDFLoader}

def init_session_state():
    logger.info("Initializing session state.")
    keys_to_init = ['student_llm', 'teacher_llm', 'prompts', 'embeddings', 'retriever', 'raw_docs', 'current_state', 'run_phase', 'messages']
    for key in keys_to_init:
        if key not in st.session_state:
            st.session_state[key] = None

@st.cache_resource
def load_llms_and_embeddings():
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
    logger.info("Loading prompts from %s", PROMPTS_FILE)
    with open(PROMPTS_FILE, 'r') as f: prompts = yaml.safe_load(f)
    logger.info("Prompts loaded successfully.")
    return prompts

@st.cache_resource
def process_uploaded_files(_files, _embeds):
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
    Args:
        doc_name (str): The exact name of the document to retrieve, as found in a [Source: ...] tag.
    """
    logger.info("TOOL CALL: citation_retriever(doc_name='%s')", doc_name)
    raw_docs = st.session_state.get('raw_docs', {})
    content = raw_docs.get(doc_name, f"Error: Document '{doc_name}' not found.")
    if "Error" in content: logger.warning("citation_retriever failed to find document: %s", doc_name)
    return content

def save_artifact(state: GraphState):
    logger.debug("Attempting to save artifact...")
    output_content = state.get('output')
    if not output_content or not isinstance(output_content, str) or not output_content.strip():
        logger.warning("Save artifact called but output was empty or invalid. Nothing will be saved.")
        st.warning("Could not save artifact: Final report content is empty.")
        return
    logger.debug(f"Artifact content (first 200 chars): {output_content[:200]}")
    try:
        filename = f"agent_output_{state.get('run_phase', 'unknown')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path = os.path.join(ARTIFACT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f: f.write(output_content)
        st.success(f"Artifact saved to `{path}`")
        logger.info("Successfully saved artifact to %s", path)
    except Exception as e:
        st.error(f"Failed to save artifact: {e}"); logger.error("Failed to save artifact: %s", e, exc_info=True)

def display_plan(plan):
    st.subheader("📝 Proposed Plan")
    if plan and "title" in plan and "plan_items" in plan:
        st.markdown(f"**Title:** {plan['title']}")
        for item in plan.get('plan_items', []): st.markdown(f"- **{item['section_id']} {item['title']}:** {item['description']}")
    else:
        st.error("Failed to generate a valid plan."); st.json(plan)

def display_evaluation(evaluation):
    st.subheader("Vetting Report")
    if not evaluation or "error" in evaluation:
        st.error("Failed to generate a valid evaluation."); st.json(evaluation); return
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

def main():
    st.set_page_config(page_title="Professional Agent System v2", layout="wide")
    init_session_state()
    
    if st.sidebar.button("Start New Session", type="primary"):
        # Clear log file for new session
        if 'log_file' in st.session_state and os.path.exists(st.session_state['log_file']):
            with open(st.session_state['log_file'], "w"):
                pass
        for key in st.session_state.keys():
            st.session_state[key] = None
        st.rerun()

    logger.info("--- Streamlit App Started / Reran ---")

    if not st.session_state.student_llm:
        st.session_state.student_llm, st.session_state.teacher_llm, st.session_state.embeddings = load_llms_and_embeddings()
        st.session_state.prompts = load_prompts()
    
    with st.sidebar:
        st.title("🔬 Agent Setup")
        uploaded_files = st.file_uploader("1. Upload Knowledge Base", type=list(DOC_LOADERS.keys()), accept_multiple_files=True)
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing documents..."):
                st.session_state.retriever, st.session_state.raw_docs = process_uploaded_files(tuple(uploaded_files), st.session_state.embeddings)
            if st.session_state.retriever: st.success("Knowledge base ready.")
            else: st.warning("No files processed or failed to build retriever.")
        st.markdown("---")
        st.info(f"Logs are in: `{os.path.abspath(log_file)}`")
        st.info(f"Outputs are in: `{os.path.abspath(ARTIFACT_DIR)}`")

    st.title("Professional Grade Research Agent (v2.3)")

    if st.session_state.retriever:
        current_phase = st.session_state.get('run_phase')
        if current_phase is None:
            st.header("Phase 1: Planning")
            if st.button("📄 Generate Research Report", use_container_width=True):
                st.session_state.run_request = st.session_state.prompts['research_generation_task']
            
            if st.session_state.get("run_request"):
                user_prompt = st.session_state.pop("run_request")
                st.session_state.current_state = {"user_prompt": user_prompt, "log": [], "documents": [], "messages": []}
                logger.info("Starting PLANNING phase for prompt: %s", user_prompt[:100])
                with st.spinner("Student agent is generating a plan..."):
                    planning_graph = build_planning_graph(st.session_state.student_llm, st.session_state.retriever, st.session_state.prompts)
                    final_plan_state = planning_graph.invoke(st.session_state.current_state)
                    st.session_state.current_state.update(final_plan_state)
                st.session_state.run_phase = "PLAN_APPROVAL"
                st.rerun()
        
        elif current_phase == "PLAN_APPROVAL":
            st.header("Phase 1: Plan Approval (HITL)")
            display_plan(st.session_state.current_state.get('plan'))
            if st.button("✅ Approve Plan & Begin Synthesis", use_container_width=True):
                st.session_state.run_phase = "SYNTHESIS"; logger.info("Plan approved. Moving to SYNTHESIS."); st.rerun()
        
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

        elif current_phase == "FINAL_REVIEW":
            st.header("Phase 4: Final Review (HITL)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Final Report Draft:**"); st.markdown(st.session_state.current_state.get('output', ''))
            with col2:
                display_evaluation(st.session_state.current_state.get('evaluation_report'))
            st.markdown("---")
            if st.button("Accept Final Report", use_container_width=True):
                st.session_state.current_state['run_phase'] = "research_report"
                save_artifact(st.session_state.current_state)
                st.success("Process complete! Artifact saved.")
                logger.info("Process complete. Final artifact accepted by user.")
                st.session_state.run_phase = None
                st.session_state.current_state = None
            
    else:
        st.info("Please upload and build a knowledge base to begin.")

if __name__ == "__main__":
    main()

