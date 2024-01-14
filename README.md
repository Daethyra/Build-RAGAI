# Build-RAGAI

## Description
This project seeks to teach you how to build Python applications with generative AI functionality by using the LangChain and Transformers libraries.

While there is a section for [OpenAI](./src/opai/), most of the code that previously existed there has been repurposed and integrated with either the [LangChain](./src/langchain/) or [Transformers](./src/transformers/) libraries. This project includes code snippets, end2end examples, and jupyter notebooks that you can augment, copy, or learn from respectively.

If you're new to building AI-powered applications, I suggest you start by playing with and executing the code in the [LangChain notebooks](./src/langchain/notebooks/). Seeing the code in action, editing it yourself, and creatively brainstorming new ideas is the best way to learn.

## Getting Started

### Installation

#### Local Code Execution and Testing
This project is developed using [PDM](https://pdm.fming.dev/). I recommend you install and use PDM to ensure you're taking steps that have already been tested and verified as working correctly. You can install PDM using `pip`:

Start by navigating to the root directory of this project.

```bash
pip install -U pdm
```

Then you'll need to install the dependencies using PDM:

```bash
pdm install
```

This command will create a virtual environment in `.venv` and install the dependencies in that environment. If you're on macOS or Linux, you can run `source .venv/bin/activate` to activate the environment. Otherwise, you can run the command `.venv/Scripts/activate` or `.venv/Scripts/activate.ps1` to activate the environment.

By using a virtual environment we avoid cross contaminating our global Python environment.

Once our virtual environment is set up we need to select it as our kernel for the Jupyter Notebook. If you're in VSCode, you can do this at the top right of the notebook. If you're using a different IDE, you'll need to look for setup help online.

When selecting the kernel, ensure you choose the one that's located inside of the `.venv` directory, and not the global Python environment.

#### Google Colab Execution and Testing

To get started in Google Colab, you can upload any of the notebooks from this repository, OR click the badge below to open the [starter LangChain RAG notebook](./src/langchain/notebooks/learn_rag.ipynb "Starter RAG Notebook for learning") in Colab.

<a target="_blank" href="https://colab.research.google.com/github/Daethyra/Build-RAGAI/blob/master/src/langchain/notebooks/learn_rag.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 'learn_rag.ipynb' In Colab"/>
</a>

---

## [LICENSE - GNU Affero GPL](./LICENSE)
