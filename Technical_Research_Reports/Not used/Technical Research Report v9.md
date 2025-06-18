# **Technical Research Report: A Tri-Modal Constitutional Agent System**

Version: 9.0

Date: 2025-06-12

## **1\. Abstract**

This document presents the formal architecture for a tri-modal, constitutional, multi-agent AI system. This version extends the previous "committee of experts" framework by introducing a third distinct workflow for generating Technical Code Documentation. The system can now dynamically route user requests to one of three specialized paths: high-level Research Report generation, implementation-focused Code Generation, or the creation of pragmatic Technical Documentation that bridges the gap between concept and implementation. The entire system continues to operate under a Shared System Prompt (Constitution), ensuring that all agent interactions—generation, critique, and refinement—are aligned toward a common set of quality standards and collaborative principles. This tri-modal capability enhances the agent's versatility and its utility as a comprehensive tool for a full research and development lifecycle.

## **2\. The Tri-Modal Architecture**

The failure of the v8 system to generate multiple code files highlighted a key limitation: the need for more specialized and explicit workflows. The v9 architecture addresses this by expanding the router's capabilities and introducing a new, dedicated generation path.

### **2.1. The Three Workflows**

The agent's supervisor can now route a user's prompt to one of three distinct computational graphs:

1. Research Report Path: Unchanged from v8. This path is optimized for producing high-level, conceptual, and implementation-agnostic reports from first principles.  
2. Technical Documentation Path: This is the new workflow. Its goal is to analyze source code and technical reports from the knowledge base to produce detailed, human-readable documentation in Markdown format. This documentation explains the code's architecture, key functions, and usage. This path includes its own generate\_documentation \-\> reflect\_on\_documentation refinement loop.  
3. Code Generation Path: This path is now specifically refined to handle the generation of multiple, distinct code files. It does so through a more directive prompting strategy that instructs the LLM to produce clearly demarcated code blocks for each required file.

### **2.2. State and Node Enhancements**

* State Definition (GraphState): The state definition remains the same, but the task\_mode can now take on a third value: documentation.  
* New Nodes:  
  * f\_generate\_documentation(S): A new node that prompts an LLM to act as a technical writer, analyzing code and reports to generate Markdown documentation.  
  * f\_reflect\_on\_documentation(S): A new reflection node that allows an agent to critique the generated documentation for clarity, accuracy, and completeness, creating a refinement loop for this new workflow.

### **2.3. Enhanced Code Generation Protocol**

To address the failure to produce multiple files, the f\_generate\_code node is now guided by a more robust prompting strategy. The prompt explicitly instructs the LLM to return a single response containing clearly labeled and separated code blocks for each required file (e.g., \#\#\# AGENT\_CORE.PY \#\#\# and \#\#\# SUPERVISOR.PY \#\#\#). The supervisor is then responsible for parsing this single output and saving the content into separate files.

### **2.4. Artifact Persistence**

A new function, save\_artifact, is added to the supervisor's workflow. Once the constitutional agents reach a consensus on a final artifact, this function automatically saves it to a local file. The filename includes the artifact type and a timestamp (e.g., agent\_output\_report\_20250612\_143000.md), and the file content includes a corresponding header.

This tri-modal architecture creates a more versatile and powerful system that can support a researcher not just with high-level theory and low-level code, but also with the crucial connective tissue of technical documentation.

