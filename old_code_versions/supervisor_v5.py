# supervisor_v5.py
# Implements TDS v6.0: Saves the new design_synthesis.json artifact.
# Updated to v5 to reflect major architectural changes.

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

# --- v5: Import from the new agent core ---
from agent_core_v5 import GraphState, build_planning_graph, build_synthesis_graph, build_evaluation_graph, build_design_agent_graph

# --- Robust, Session-Based Logger Setup ---
def setup_logger():
    logger_name = "ResearchAgentV5"
    if 'logger' not in st.session_state or st.session_state.logger.name != logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            log_file = "agent_run_v5.log"
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

# --- v5: Update constants for new file names ---
ARTIFACT_DIR, PROMPTS_FILE = "agent_outputs_v5", "prompts_v5.yaml"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
DOC_LOADERS = {".txt": TextLoader, ".md": TextLoader, ".py": TextLoader, ".docx": UnstructuredWordDocumentLoader, ".pdf": PyPDFLoader}

def init_session_state():
    keys_to_init = ['student_llm', 'teacher_llm', 'prompts', 'embeddings', 'retriever', 'raw_docs', 'current_state', 'run_phase']
    for key in keys_to_init:
        st.session_state.setdefault(key, None)

@st.cache_resource
def load_llms_and_embeddings():
    student_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
    teacher_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return student_llm, teacher_llm, embeddings

@st.cache_data
def load_prompts():
    with open(PROMPTS_FILE, 'r') as f: return yaml.safe_load(f)

@st.cache_resource
def process_uploaded_files(_files, _embeds):
    if not _files: return None, None
    docs, raw_docs = [], {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for f in _files:
            path = os.path.join(temp_dir, f.name)
            with open(path, "wb") as out: out.write(f.getbuffer())
            if loader_cls := DOC_LOADERS.get(os.path.splitext(f.name)[1]):
                try:
                    loader = loader_cls(path)
                    loaded_docs = loader.load(); [d.metadata.update({'source': f.name}) for d in loaded_docs]
                    docs.extend(loaded_docs)
                    raw_docs[f.name] = "\n".join([d.page_content for d in loaded_docs])
                except Exception as e:
                    logger.error("Failed to load file %s: %s", f.name, e)
    if not docs: return None, None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return FAISS.from_documents(text_splitter.split_documents(docs), _embeds).as_retriever(), raw_docs

@tool
def citation_retriever(doc_name: str) -> str:
    """Retrieves the full text content of a source document to verify claims."""
    return st.session_state.get('raw_docs', {}).get(doc_name, f"Error: Document '{doc_name}' not found.")

def save_artifacts(state: GraphState, run_phase: str):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def save_file(content, extension):
        if content and isinstance(content, (str, dict)):
            try:
                filename = f"agent_output_{run_phase}_{timestamp}.{extension}"
                path = os.path.join(ARTIFACT_DIR, filename)
                with open(path, "w", encoding="utf-8") as f:
                    if isinstance(content, dict): json.dump(content, f, indent=2)
                    else: f.write(content)
                st.success(f"Artifact saved: `{path}`")
                logger.info("Saved artifact to %s", path)
            except Exception as e:
                st.error(f"Failed to save .{extension} artifact: {e}")
                logger.error("Failed to save artifact: %s", e, exc_info=True)

    if run_phase == 'research_report':
        save_file(state.get('output'), 'md')
        save_file(state.get('a2a_output'), 'json')
    elif run_phase == 'design_spec':
        save_file(state.get('design_spec'), 'md')
        save_file(state.get('design_synthesis_json'), 'json')

# --- UI Display Functions ---
def display_plan(plan):
    st.subheader("📝 Proposed Plan")
    if plan and "title" in plan and "plan_items" in plan:
        st.markdown(f"**Title:** {plan['title']}")
        for item in plan.get('plan_items', []): st.markdown(f"- **{item['section_id']} {item['title']}:** {item['description']}")
    else: st.error("Failed to generate a valid plan."); st.json(plan or {})

def display_evaluation(evaluation):
    st.subheader("Vetting Report")
    if not evaluation or "error" in evaluation:
        st.error("Failed to generate valid evaluation."); st.json(evaluation or {}); return
    st.metric("Consistency Score", f"{evaluation.get('overall_consistency_score', 0.0):.2f}")
    st.info(f"**Notes:** {evaluation.get('consistency_notes', 'N/A')}")
    goal_status = evaluation.get('goal_alignment_check', 'FAIL')
    st.success(f"**Goal Alignment:** {goal_status} - {evaluation.get('goal_alignment_notes', '')}") if goal_status == 'PASS' else st.error(f"**Goal Alignment:** {goal_status} - {evaluation.get('goal_alignment_notes', '')}")
    st.markdown("**Citation Audit:**")
    for audit in evaluation.get('citation_audit', []):
        status = audit.get('verification_status', 'FAIL'); claim = audit.get('claim', 'N/A'); tag = audit.get('citation_tag', 'N/A')
        st.write(f"✅ **PASS:** {claim}" if status == 'PASS' else f"❌ **FAIL:** {claim} (Cited: {tag})")

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Agentic System v5", layout="wide")
    init_session_state()
    
    with st.sidebar:
        st.title("🔬 Agent Setup (v5)")
        if st.button("Start New Session", type="primary"): st.session_state.clear(); st.rerun()
        uploaded_files = st.file_uploader("1. Upload Knowledge Base", type=list(DOC_LOADERS.keys()), accept_multiple_files=True)
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing documents..."):
                st.session_state.retriever, st.session_state.raw_docs = process_uploaded_files(tuple(uploaded_files), st.session_state.embeddings)
            if st.session_state.retriever: st.success("Knowledge base ready.")
        st.markdown("---"); st.info(f"Logs: `{os.path.abspath(log_file)}`"); st.info(f"Outputs: `{os.path.abspath(ARTIFACT_DIR)}`")

    st.title("Autonomous R&D Agent (v5.0)")
    
    if 'student_llm' not in st.session_state or st.session_state.student_llm is None:
        st.session_state.student_llm, st.session_state.teacher_llm, st.session_state.embeddings = load_llms_and_embeddings()
        st.session_state.prompts = load_prompts()

    if not st.session_state.get('retriever'):
        st.info("Please upload documents and build a knowledge base to begin."); return

    current_phase = st.session_state.get('run_phase')

    # PHASE 1: Research Planning
    if current_phase is None:
        st.header("Phase 1: Research")
        if st.button("📄 Generate Research Report", use_container_width=True):
            user_prompt = st.session_state.prompts['research_generation_task']
            st.session_state.current_state = GraphState(user_prompt=user_prompt, messages=[])
            with st.spinner("Agent is generating a plan..."):
                final_plan_state = build_planning_graph(st.session_state.student_llm, st.session_state.retriever, st.session_state.prompts).invoke(st.session_state.current_state)
                st.session_state.current_state.update(final_plan_state)
            st.session_state.run_phase = "PLAN_APPROVAL"; st.rerun()
    
    # HITL 1: Plan Approval
    elif current_phase == "PLAN_APPROVAL":
        st.header("Phase 1: Plan Approval (HITL)")
        display_plan(st.session_state.current_state.get('plan'))
        if st.button("✅ Approve Plan & Begin Synthesis", use_container_width=True):
            st.session_state.run_phase = "SYNTHESIS"; st.rerun()
    
    # PHASE 2: Research Synthesis
    elif current_phase == "SYNTHESIS":
        st.header("Phase 2: Synthesis")
        with st.spinner("Agent is generating the report..."):
            st.session_state.current_state.update({"working_memory": {}, "completed_plan_items": []})
            final_synth_state = build_synthesis_graph(st.session_state.student_llm, st.session_state.prompts).invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_synth_state)
        st.session_state.run_phase = "EVALUATION"; st.rerun()

    # PHASE 3: Research Evaluation
    elif current_phase == "EVALUATION":
        st.header("Phase 3: Evaluation")
        st.markdown(st.session_state.current_state.get('output', ''))
        with st.spinner("Agent is auditing the report..."):
            final_eval_state = build_evaluation_graph(st.session_state.teacher_llm, st.session_state.prompts, [citation_retriever]).invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_eval_state)
        st.session_state.run_phase = "FINAL_REVIEW"; st.rerun()

    # HITL 2: Final Review & Next Step
    elif current_phase == "FINAL_REVIEW":
        st.header("Phase 4: Final Review & Next Step (HITL)")
        col1, col2 = st.columns(2); col1.markdown("**Final Report Draft:**"); col1.markdown(st.session_state.current_state.get('output', ''))
        with col2: display_evaluation(st.session_state.current_state.get('evaluation_report'))
        with st.expander("View Agent-to-Agent (A2A) JSON Artifact"): st.json(st.session_state.current_state.get('a2a_output', {}))
        
        st.markdown("---"); col_a, col_b = st.columns(2)
        if col_a.button("Accept & Save Research Artifacts", use_container_width=True):
            save_artifacts(st.session_state.current_state, 'research_report')
        if col_b.button("▶️ Generate Design Specification", use_container_width=True, type="primary"):
            st.session_state.run_phase = "DESIGN_GENERATION"; st.rerun()
        with st.expander("Or, provide feedback and restart"):
            feedback = st.text_area("Your feedback:")
            if st.button("Re-Plan with This Feedback"):
                st.session_state.current_state = GraphState(user_prompt=st.session_state.current_state['user_prompt'], user_feedback=feedback, evaluation_report=st.session_state.current_state.get('evaluation_report'), messages=[])
                st.session_state.run_phase = None; st.rerun()
                
    # PHASE 5: DESIGN GENERATION
    elif current_phase == "DESIGN_GENERATION":
        st.header("Phase 5: Design Specification Generation")
        with st.spinner("Design Agent is generating the technical specification..."):
            final_design_state = build_design_agent_graph(st.session_state.student_llm, st.session_state.prompts).invoke(st.session_state.current_state)
            st.session_state.current_state.update(final_design_state)
        st.session_state.run_phase = "DESIGN_REVIEW"; st.rerun()

    # HITL 3: Design Review
    elif current_phase == "DESIGN_REVIEW":
        st.header("Phase 6: Design Specification Review (HITL)")
        st.markdown(st.session_state.current_state.get('design_spec', "# No Design Spec Generated"))
        with st.expander("View Design Synthesis JSON Artifact"): st.json(st.session_state.current_state.get('design_synthesis_json', {}))
        
        st.markdown("---")
        if st.button("Accept & Save Design Artifacts", use_container_width=True):
            save_artifacts(st.session_state.current_state, 'design_spec')
            st.info("Next step: Implement Coding Agent. This functionality will be in v6.")
        if st.button("Start Over"):
            st.session_state.clear(); st.rerun()

if __name__ == "__main__":
    main()

