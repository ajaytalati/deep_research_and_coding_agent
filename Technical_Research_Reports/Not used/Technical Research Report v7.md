# **Technical Research Report: A Constitutional, Multi-Agent System for Robust Generation**

Version: 7.0

Date: 2025-06-11

## **1\. Abstract**

This document presents the architecture for a constitutional, multi-agent system designed to achieve a higher degree of logical and semantic robustness in generated artifacts. This architecture evolves from a single, multi-path agent to a collaborative-adversarial system comprising two distinct but identical agents, each powered by a different Large Language Model (e.g., Gemini, Deepseek). The system operates on a principle of "constitutional AI," where the agents engage in a process of mutual critique and iterative refinement. One agent generates a draft (either a technical report or source code), which is then passed to the second agent for critical review. This cross-pollination of critique continues until a consensus state is reached, at which point a single, validated artifact is presented to the user. This multi-agent approach directly addresses the challenge of identifying and correcting subtle logical errors, moving beyond simple syntactic validation to a more sophisticated form of automated peer review.

## **2\. System Architecture: A Committee of Agents**

The core of the v7 architecture is a move from a single-agent graph to a supervised, multi-agent system. This system is composed of three main components: a reusable Agent Core library, a Multi-Agent Supervisor, and two or more Agent Instances.

### **2.1. The Agent Core Library (**agent\_core.py**)**

To facilitate a multi-agent design, the agent's fundamental logic is refactored into a modular, reusable library. This library is designed to be LLM-agnostic.

* Components:  
  * GraphState Definition: The standardized TypedDict defining the agent's state (prompt, documents, output, critique, etc.).  
  * Node Functions: A collection of Python functions that implement the core operations of the graph: retrieve\_context, route\_task, generate\_report, reflect\_on\_report, generate\_code, and test\_code. These functions are designed to accept an LLM client object as an argument, allowing them to be used by any agent instance.  
  * Graph Builder: A factory function, build\_agent\_graph(), that takes an LLM client and assembles a complete LangGraph agent instance from the node functions.

### **2.2. The Multi-Agent Supervisor (**multi\_agent\_supervisor.py**)**

The supervisor acts as the orchestrator or "project manager" of the system. It is responsible for initializing the agents and managing the constitutional workflow.

* Responsibilities:  
  * Initialization: Instantiates multiple LLM clients (e.g., a Gemini client, a Deepseek client).  
  * Graph Instantiation: Uses the build\_agent\_graph() factory from the agent\_core library to create two or more independent agent graph instances, each linked to a different LLM.  
  * Workflow Orchestration: Manages the flow of data and tasks between the agents.  
  * UI Management: Controls the Streamlit user interface, presenting the overall progress and the final, converged output.

### **2.3. The Constitutional Workflow**

The supervisor executes a "constitutional" loop designed to achieve a high-quality, validated consensus.

1. Initial Generation (Parallel):  
   * The user's prompt is passed to both Agent A (Gemini) and Agent B (Deepseek).  
   * Both agents run their internal generate \-\> reflect/test loops to produce their own initial "best effort" draft of the requested artifact (report or code).  
2. Cross-Critique Cycle (Iterative):  
   * Step 2a (B critiques A): The supervisor takes the draft from Agent A and passes it to Agent B. Agent B is invoked with a specific "critique" prompt: *"You are a critical reviewer. Analyze the following artifact generated by another AI agent. Identify any logical fallacies, semantic ambiguities, coding errors, or potential improvements."*  
   * Step 2b (A critiques B): Symmetrically, the draft from Agent B is passed to Agent A for critique.  
   * Step 2c (Self-Refinement with External Critique): The critique generated by Agent B is fed back into Agent A's refinement loop. Agent A now has both its own self-reflection *and* the external critique to guide its next revision. The same occurs for Agent B.  
3. Convergence Check:  
   * After each cross-critique and refinement cycle, the supervisor checks for convergence. Convergence is reached when the critiques generated by both agents become minimal or empty (e.g., they both respond with "No further critiques").  
4. Final Output:  
   * Once convergence is reached, the supervisor selects one of the artifacts (e.g., from the agent that performed the last successful refinement) and presents it to the user as the single, validated output.

This architecture transforms the process from a single agent's attempt into a structured, adversarial collaboration that more closely mimics how human expert teams work to produce high-quality, error-free results.

