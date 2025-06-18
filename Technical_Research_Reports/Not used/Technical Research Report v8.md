# **Technical Research Report: A Constitutional, Multi-Agent System with a Shared System Prompt**

Version: 8.0

Date: 2025-06-12

## **1\. Abstract**

This document presents the architecture for a constitutional, multi-agent AI system designed to overcome the limitations of unaligned adversarial critique and produce high-fidelity, logically coherent artifacts. This version evolves from a simple multi-agent loop to a sophisticated "committee of experts" framework. The core innovation is the introduction of a Shared System Prompt, or "Constitution," that governs the behavior and objectives of all participating agents. By providing a common set of principles, goals, and rules of engagement, the agents are aligned to act as collaborators rather than adversaries. The workflow involves each agent generating an initial draft, followed by iterative rounds of cross-critique and refinement, where all critiques are explicitly framed by the shared constitution. This process ensures that the agents converge on a single, validated output that holistically satisfies the user's prompt while adhering to predefined quality standards.

## **2\. The Failure of Unaligned Critique and the Need for a Constitution**

The previous architecture (v7), while implementing a multi-agent cross-review loop, revealed a critical failure mode. When instructed to simply "critique" each other's work without a shared goal, the agents entered a destructive adversarial loop. This resulted in a "race to the bottom," where agents would focus on minor or pedantic flaws, leading to the removal of valid information and a degradation of the final output's quality and completeness. The agents optimized for "finding a flaw" rather than "collaboratively achieving the user's goal."

This observation necessitates a shift from a simple adversarial review to a structured, constitutional debate. The v8 architecture addresses this by introducing a Shared System Prompt that acts as a constitution for the agent committee.

## **3\. The Constitutional Architecture**

The system retains the modular design of a central Supervisor orchestrating multiple Agent Instances built from a common Agent Core library. The key addition is the CONSTITUTIONAL\_PROMPT.

### **3.1. The Shared System Prompt (The "Constitution")**

This is a detailed, high-level prompt that is injected into the workflow at critical stages. It defines the "rules of the game" for all participating agents.

* Core Components of the Constitution:  
  * Shared Objective: Clearly defines the ultimate goal (e.g., "Produce the most comprehensive, accurate, and logically consistent technical report that synthesizes all provided versions of the source material.").  
  * Guiding Principles: Outlines the desired qualities of the output (e.g., "Prioritize holistic synthesis over pedantic error-finding," "Ensure the evolution of concepts is captured," "The final output must be self-contained and functional.").  
  * Rules of Critique: Instructs agents on how to perform reviews. Critiques must be constructive, actionable, and explicitly tied to the shared objective and guiding principles.  
  * Role Definition: Defines the role of each agent as a "collaborative expert" working towards a common goal.

### **3.2. The Constitutional Workflow**

The supervisor manages an enhanced workflow that leverages this shared constitution.

1. Initial Generation Under the Constitution:  
   * The user's prompt is combined with the Shared System Prompt.  
   * This combined meta-prompt is given to both Agent A and Agent B to guide their initial draft generation. This ensures both agents start with the same high-level goals.  
2. Constitutional Cross-Critique Cycle:  
   * Step 2a (B critiques A): The supervisor takes Agent A's draft. It then creates a new prompt for Agent B that includes:  
     1. The Shared System Prompt ("Here are our rules...").  
     2. Agent A's draft ("Here is the artifact to review...").  
     3. A specific instruction ("Based on our shared constitution, critique this artifact. Does it meet our objective?").  
   * Step 2b (A critiques B): The same process is repeated, with Agent A critiquing Agent B's work based on the constitution.  
3. Constitutional Refinement:  
   * The constitutionally-grounded critique from Agent B is passed back to Agent A.  
   * Agent A is prompted to refine its draft, with the explicit goal of better satisfying the shared constitution in light of the provided critique.  
4. Convergence and Finalization:  
   * The loop continues until the constitutional critiques become minimal or empty (e.g., both agents respond with "This artifact aligns with our constitution. No further critique needed.").  
   * The supervisor then presents the final, converged artifact to the user.

This constitutional approach transforms the multi-agent dynamic from a zero-sum game of finding flaws into a positive-sum collaboration aimed at producing a superior, aligned output.

