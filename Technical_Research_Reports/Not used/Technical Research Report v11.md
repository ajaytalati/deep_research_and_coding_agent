# **Technical Research Report: An Agentic System with Externalized Prompt Management**

Version: 11.0

Date: 2025-06-13

## **1\. Abstract**

This document details a critical architectural refactoring of the tri-modal constitutional agent system, designed to enhance modularity, maintainability, and user-configurability. Previous versions embedded complex, multi-line prompts directly within the supervisor's Python script, a practice that proved to be inflexible and cumbersome. The v11 architecture addresses this flaw by externalizing all workflow prompts into a dedicated prompts.yaml configuration file. This strategic separation of configuration (prompts) from logic (the Python code) allows for easier experimentation and refinement of agent instructions without modifying the core application. The supervisor is now responsible for loading these external prompt templates at runtime and dynamically populating them with the necessary context from the sequential workflow, representing a more mature and robust software design pattern.

## **2\. The Problem with Hard-Coded Prompts**

The previous architecture, while functionally capable, suffered from a significant design flaw: the embedding of large, complex prompt strings directly within the supervisor.py script. This led to several issues:

* Reduced Readability: The application logic was cluttered with large blocks of text, making the code harder to follow.  
* Poor Maintainability: To edit or tune a prompt, a user would need to directly modify the Python source code, increasing the risk of introducing bugs.  
* Lack of Flexibility: Experimenting with different prompting strategies was a cumbersome process of finding and replacing strings within the script.

## **3\. The v11 Solution: Externalized Prompt Configuration**

The v11 architecture solves this problem by adopting a standard software engineering principle: the separation of configuration from code.

* prompts.yaml: A new configuration file is introduced. This file uses the YAML format, which is highly readable and ideal for managing multi-line strings. It contains all the prompts for the different stages of the agent's workflows (report generation, code generation, documentation generation). Each prompt is given a unique key for easy retrieval.  
* Dynamic Prompt Loading: The supervisor.py script is updated to include logic for reading and parsing prompts.yaml upon startup. It loads all prompt templates into a dictionary.  
* Context Injection: When a workflow is executed, the supervisor retrieves the appropriate template from the loaded dictionary and uses standard string formatting (e.g., prompt\_template.format(...)) to inject the dynamic context (such as the content of a previously generated report or code file) before passing the final, complete prompt to the agent system.

This design makes the system significantly more robust, maintainable, and user-friendly for experimentation.

