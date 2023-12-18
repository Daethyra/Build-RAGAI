# LLM Utilikit

## Description
The Utilikit is a Python library designed to enhance large-language-model projects. It offers a variety of components, prompts, and templates, and is open for contributions from users. The library aims to provide a quick start for new projects and modular, reusable components for existing ones. 

---
<s>
## Getting Started

This repository has a split purpose but a sole focus.

### Purpose 1: *Support users of all levels with power prompts.*

* The Utilikit features two main types of prompts:
  * [Multi-shot](./prompts_MASTER.md#Multi-Shot-Prompts) and [user-role](./prompts_MASTER.md#User-Role-Prompts), detailed in the [prompts_MASTER.md](./prompts_MASTER.md) file.
    * The prompt examples are meant to be a starting point, and most, if not all, prompts require tweaking or filling in blanks in order to prove useful. 
* If you're confident in prompting large language models and are looking for templates you'll likely gain the most value from the [prompt-cheatsheet.](./prompt-cheatsheet.md)

### Purpose 2: *Provide prebuilt Python componenets to help developers of all levels quickly get started with support libraries or augment LLM related projects.*

* [Supported Libraries](./src/llm_utilikit/):
  * [OpenAI](./src/llm_utilikit/OpenAI/)
  * [LangChain](./src/llm_utilikit/LangChain/)
  * [HuggingFace](./src/llm_utilikit/HuggingFace/)

### Purpose 3: *Distribute Reference Documents for Developers and AI Programming Assistants.*

[<ins>Here</ins>](./Custom-GPT-Uploadable_Knowledge_Base/) you'll find reference documents for building Python applications, specifically those using large-language-models as their base. 

- The Custom GPT, [Assistant Architect](https://chat.openai.com/g/g-gOeFNMJ8Z-assistant-architect-aa4llm), is built on these markdown [files](./Custom-GPT-Uploadable_Knowledge_Base/Assistant_Architect/).
</s>
---

## Usage Guide

1. Decide which libraries you'd like to build your project with.
2. Try to find prebuilt Python modules for your own project.
3. Modify code logic as necessary.

You may wish to use PDM, as I do, to install the project via:
```
cd llm-utilikit
pip install -U pdm
pdm install
```

Otherwise, install all libraries with the following command:
```
pip install .
```

---

## [LICENSE - GNU Affero GPL](./LICENSE)
