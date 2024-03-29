# LangSmith: a platform for building production-grade LLM applications
LangSmith is an innovative and dynamic testing framework for evaluating language models and AI applications. As a platform, it is capable of building production-grade LLM applications. In the realm of language model testing, LangSmith emerges as a robust and versatile testing framework.

It lets you debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework and seamlessly integrates with LangChain, the go-to open source framework for building with LLMs.
- LangSmith is developed by LangChain, the company behind the open source LangChain framework.

## Get started: Installation and Enable tracing
If you already use LangChain, you can connect to LangSmith in a few steps:

1. Create a LangSmith account using one of the supported login methods.
2. Create an API Key by navigating to the settings page.
3. Install the latest version LangChain for your target environment and programming language.
- `pip install -U langchain`
4. Configure runtime environment:
* Replace "your-api-key" with the API key generated in step 1
* Replace "your-openai-api-key" with an OpenAI API Key from here

The LangSmith SDK is automatically installed by LangChain. If not using LangChain, install with:
- `pip install -U langsmith`

##### Enable LangSmith tracing via **setting environment variables**
First, configure your environment variables to tell LangChain to log traces. This is done by setting the `LANGCHAIN_TRACING_V2` environment variable to true. You can tell LangChain which project to log to by setting the `LANGCHAIN_PROJECT` environment variable (if this isn’t set, runs will be logged to the default project). This will automatically create the project for you if it doesn’t exist. You must also set the `LANGCHAIN_ENDPOINT` and `LANGCHAIN_API_KEY` environment variables.
```python
# Notice the subtle differences in the env-vars below, as there are multiple endpoints being set.
# Required variables; Always set the following 2 variables
import os
os.environ["LANGCHAIN_API_KEY"] = "<your api key>"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Optional variables depending on current use case
os.environ["OPENAI_API_KEY"] = "sk-" # Update with your OpenAI API key
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" # Leave unchanged unless using a hosted instance of LangSmith.
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"  # Example project name. Defaults to "default"
```
##### Trace your first run
5. Run the example code below:
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")
```
Congratulations! Your first run has been traced and is now visible in LangSmith! In this short example, we initialize LangSmith tracing via environment variables and run `invoke` of the LangChain library as normal. 
Navigate to the [projects page](https://smith.langchain.com/projects) to view your "Hello, world!" trace.

## Overview: How LangSmith works under the hood.
LangSmith helps you visualize, debug, and improve your LLM apps. This section reviews some functionality LangSmith provides around logging and tracing.

### Log Run
LLM applications can get complicated quickly, especially if you are building working with agents or complex chains where there could be many layers of LLMs and other components. LangSmith makes it easy to log runs of your LLM applications so you can inspect the inputs and outputs of each component in the chain. This is especially useful when you are trying to debug your application or understand how a given component is behaving. This is done in two steps.

### Organize your work
Runs are saved to projects. Root level runs are called `traces` and are saved to the `default` project if you don't specify a project name. You can also view all of your runs un-nested. You can create as many projects as you like to help you organize your work by context. For instance, you might have a project for each of your production LLM application environments, or you might have a project to separate runs on different days. You can also create projects for specific experiments or debugging sessions.

### Visualize your Runs
Every time you run a LangChain component with tracing enabled or use the LangSmith SDK to save run trees directly, the call hierarchy for the run is saved and can be visualized in the app. You can drill down into the components inputs and outputs, invocation parameters, response time, feedback, token usage, and other important information to help you inspect your run. You can even rate the run to help you keep collect data for training, testing, and other analysis.

### Running in the Playground
Once you have a run trace, you can directly modify the prompts and parameters of supported chains, LLMs, and chat models to see how they impact the output. This is a great way to quickly iterate on model and prompt configurations without having to switch contexts. All playground runs are logged to a "playground" project for safe keeping.

# Tracing Integration

## Tracing Agents: What are agents, and how can we trace them?
The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In _agents_, a language model is used as a **reasoning engine** to determine which actions to take and in which order.
  - Note: The following snippet doesn't set up the tools or agents to actually be workable, but uses real logic that can be built on top of.

Since LangChain agents and agent executors are types of chains, we can trace them:
##### Trace agents, agent executors, and chains: Snippet
```python
from langchain import agents, tools


agent_executor = agents.initialize_agent(
    llm=chat_models.ChatOpenAI(),
    tools=[tools.ReadFileTool(), tools.WriteFileTool(), tools.ListDirectoryTool()],
    agent=agents.AgentType.OPENAI_FUNCTIONS,
)
with callbacks.collect_runs() as cb:
    result = agent_executor.with_config({"run_name": "File Agent"}).invoke("What files are in the current directory?")
    run = cb.traced_runs[0]
    print(result['output'])
```
- The files in the current directory are:
  1. run-naming.ipynb
  2. img
  3. .ipynb_checkpoints
```python
callbacks.tracers.langchain.wait_for_all_tracers()
print(f"Saved name: {run.name}")
```
- Saved name: File Agent
  - The resulting agent trace will reflect the custom name you've assigned to it.

## Customizing Run Names
Every LangSmith run receives a name. This name is visible in the UI and can be employed later for querying a particular run. In the context of tracing chains constructed with LangChain, the default run name is derived from the class name of the invoked object.

For runs categorized as "Chain", the name can be configured by calling the [runnable](https://python.langchain.com/docs/expression_language/ "LangChain Expression Language Documentation") object's `with_config({"run_name": "My Run Name"})` method. This guide illustrates its application through several examples.

Note: Only chains and general runnables support custom naming; LLMs, chat models, prompts, and retrievers do not.

```python
# %pip install -U langchain --quiet
import os
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" # Update with your API URL if using a hosted instance of Langsmith.
os.environ["LANGCHAIN_API_KEY"] = "YOUR API KEY" # Update with your API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
project_name = "YOUR PROJECT NAME" # Update with your project name
os.environ["LANGCHAIN_PROJECT"] = project_name # Optional: "default" is used if not set
# ---
from langsmith import Client
client = Client()
from langchain import chat_models, prompts, callbacks, schema

chain = (
    prompts.ChatPromptTemplate.from_template("Reverse the following string: {text}")
    | chat_models.ChatOpenAI()
).with_config({"run_name": "StringReverse"})

with callbacks.collect_runs() as cb:
    for chunk in chain.stream({"text": "🌑🌒🌓🌔🌕"}):
        print(chunk.content, flush=True, end="")
    run = cb.traced_runs[0]
```
If you inspect the run object on the [LangSmith webpage](https://smith.langchain.com/), you can see the run name is now "StringReverse". You can query within a project for runs with this name to see all the times this chain was used. Do so using the filter syntax `eq(name, "MyRunName")`.
`print(f"Saved name {run.name}")`
- Saved name StringReverse

```
# List with the name filter to get runs with the assigned name
next(client.list_runs(project_name=project_name, filter='eq(name, "StringReverse")'))
```
- Run(id=UUID('7fc65bbe-b3dc-401d-9bb2-e99b6cffb312'), name='StringReverse', start_time=datetime.datetime(2023, 9, 20, 21, 55, 42, 612598), run_type='chain', end_time=datetime.datetime(2023, 9, 20, 21, 55, 43, 550029), extra={'runtime': {'cpu': {'time': {'sys': 2.635597312, 'user': 1.89599424}, 'percent': 0.0, 'ctx_switches': {'voluntary': 12382.0, 'involuntary': 0.0}}, 'mem': {'rss': 98828288.0}, 'library': 'langchain', 'runtime': 'python', 'platform': 'macOS-13.5-arm64-arm-64bit', 'sdk_version': '0.0.38', 'thread_count': 35.0, 'library_version': '0.0.296', 'runtime_version': '3.11.5', 'langchain_version': '0.0.296', 'py_implementation': 'CPython'}}, error=None, serialized=None, events=[{'name': 'start', 'time': '2023-09-20T21:55:42.612598'}, {'name': 'end', 'time': '2023-09-20T21:55:43.550029'}], inputs={'text': '🌑🌒🌓🌔🌕'}, outputs={'output': {'content': '🌕🌔🌓🌒🌑', 'example': False, 'additional_kwargs': {}}}, reference_example_id=None, parent_run_id=None, tags=[], execution_order=1, session_id=UUID('41597437-b152-43e0-b05a-1cbd454d2519'), child_run_ids=[UUID('f974cb03-8878-4a86-8208-00ae940bf34a'), UUID('af466088-14c9-42bb-afe2-7a0bc766513c')], child_runs=None, feedback_stats=None, app_path='/o/9a6371ef-ea6a-4860-b3bd-9614084873e7/projects/p/41597437-b152-43e0-b05a-1cbd454d2519/r/7fc65bbe-b3dc-401d-9bb2-e99b6cffb312', manifest_id=UUID('27e257cd-cfe8-4b9c-9f52-9b9401893b59'), status='success', prompt_tokens=None, completion_tokens=None, total_tokens=None, first_token_time=None, parent_run_ids=None)

## Runnable Lambda

### Simple Chain
LangChain's [RunnableLambdas](https://api.python.langchain.com/en/latest/schema/langchain.schema.runnable.base.RunnableLambda.html#langchain.schema.runnable.base.RunnableLambda) are custom functions that can be invoked, batched, streamed, and/or transformed.

By default (in langchain versions >= 0.0.283), the name of the lambda is the function name. You can customize this by calling `with_config({"run_name": "My Run Name"})` on the runnable lambda object.

```python
def reverse_and_concat(txt: str) -> str:
    return txt[::-1] + txt

lambda_chain = chain | schema.output_parser.StrOutputParser() | reverse_and_concat
# ---
with callbacks.collect_runs() as cb:
    print(lambda_chain.invoke({"text": "🌑🌒🌓🌔🌕"}))
    # We will fetch just the lambda run (which is the last child run in this root trace)
    run = cb.traced_runs[0].child_runs[-1]

# If you are using LangChain < 0.0.283, this will be "RunnableLambda"
callbacks.tracers.langchain.wait_for_all_tracers()
print(f"Saved name: {run.name}")
```
- Saved name: reverse_and_concat

#### Customize Lambda Name
In the lambda_chain above, our function was automatically promoted to a "RunnableLambda" via the piping syntax. We can customize the run name using the with_config syntax once the object is created.

### Runnable Lambda
```python
from langchain.schema import runnable

configured_lambda_chain = chain | schema.output_parser.StrOutputParser() | runnable.RunnableLambda(reverse_and_concat).with_config({"run_name": "LambdaReverse"})
with callbacks.collect_runs() as cb:
    print(configured_lambda_chain.invoke({"text": "🌑🌒🌓🌔🌕"}))
    run = cb.traced_runs[0].child_runs[-1]
callbacks.tracers.langchain.wait_for_all_tracers()
print(f"Saved name: {run.name}")
```
- Saved name: LambdaReverse

### End-to-end Python example of Tracing Agents using LangSmith

```python
"""Example implementation of a LangChain Agent."""
import logging
from datetime import datetime
from functools import partial
import streamlit as st
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.tools.render import format_tool_to_openai_function
from langsmith import Client
from streamlit_feedback import streamlit_feedback

st.set_page_config(
    page_title="Streamlit Agent with LangSmith",
    page_icon="🦜️️🛠️",
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = Client()
st.subheader("🦜🛠️ Ask the bot some questions")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")

tools = [
    DuckDuckGoSearchResults(
        name="duck_duck_go", args_schema=DDGInput
    ),  # General internet search using DuckDuckGo
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Current date: {time}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(time=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

MEMORY = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x.get("chat_history") or [],
    }
    | prompt
    | llm.bind_functions(functions=[format_tool_to_openai_function(t) for t in tools])
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

def _submit_feedback(user_response: dict, emoji=None, run_id=None):
    score = {"👍": 1, "👎": 0}.get(user_response.get("score"))
    client.create_feedback(
        run_id=run_id,
        key=user_response["type"],
        score=score,
        comment=user_response.get("text"),
        value=user_response.get("score"),
    )
    return user_response

if st.sidebar.button("Clear message history"):
    MEMORY.clear()

feedback_kwargs = {
    "feedback_type": "thumbs",
    "optional_text_label": "Rate this response in LangSmith",
}
if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0

messages = st.session_state.get("langchain_messages", [])
for i, msg in enumerate(messages):
    avatar = "🦜" if msg.type == "ai" else None
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)
    if msg.type == "ai":
        feedback_key = f"feedback_{int(i/2)}"

        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None

        disable_with_score = (
            st.session_state[feedback_key].get("score")
            if st.session_state[feedback_key]
            else None
        )
        # This actually commits the feedback
        streamlit_feedback(
            **feedback_kwargs,
            key=feedback_key,
            disable_with_score=disable_with_score,
            on_submit=partial(
                _submit_feedback, run_id=st.session_state[f"run_{int(i/2)}"]
            ),
        )

if st.session_state.get("run_url"):
    st.markdown(
        f"View trace in [🦜🛠️ LangSmith]({st.session_state.run_url})",
        unsafe_allow_html=True,
    )
if prompt := st.chat_input(placeholder="Ask me a question!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant", avatar="🦜"):
        message_placeholder = st.empty()
        full_response = ""
        # Define the basic input structure for the chains
        input_dict = {
            "input": prompt,
        }
        input_dict.update(MEMORY.load_memory_variables({"query": prompt}))
        st_callback = StreamlitCallbackHandler(st.container())
        with tracing_v2_enabled("langsmith-streamlit-agent") as cb:
            for chunk in agent_executor.stream(
                input_dict,
                config={"tags": ["Streamlit Agent"], "callbacks": [st_callback]},
            ):
                full_response += chunk["output"]
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            feedback_kwargs = {
                "feedback_type": "thumbs",
                "optional_text_label": "Please provide extra information",
                "on_submit": _submit_feedback,
            }
            run = cb.latest_run
            MEMORY.save_context(input_dict, {"output": full_response})
            feedback_index = int(
                (len(st.session_state.get("langchain_messages", [])) - 1) / 2
            )
            st.session_state[f"run_{feedback_index}"] = run.id
            # This displays the feedback widget and saves to session state
            # It will be logged on next render
            streamlit_feedback(**feedback_kwargs, key=f"feedback_{feedback_index}")
            try:
                url = cb.get_run_url()
                st.session_state.run_url = url
                st.markdown(
                    f"View trace in [🦜🛠️ LangSmith]({url})",
                    unsafe_allow_html=True,
                )
            except Exception:
                logger.exception("Failed to get run URL.")
```

---

# LangSmith Automated Feedback for tuning LLMsAdvanced and Niche Use Cases:

## Creating an Automated Feedback Pipeline

Description: If the user feedback metrics are substantially negative, you can isolate problematic runs for debugging or fine-tuning. This tutorial shows you how to set up an automated feedback pipeline for your language models.

Steps:
1. Filter Runs: First, identify the runs you want to evaluate. For details, refer to the Run Filtering Documentation.

2. Define Feedback Logic: Create a chain or function to calculate the feedback metrics.

3. Send Feedback to LangSmith:

- Use the client.create_feedback method to send metrics.
- Alternatively, use client.evaluate_run, which both evaluates and logs metrics for you.

1. Select Runs
In this example, we will be adding model-based feedback to the run traces within a single project. To find your project name or ID, you can go to the Projects page for your organization and then call the list_runs() method on the LangSmith client.

```python
runs = client.list_runs(project_name=project_name)
# If your project is capturing logs from a deployed chain or agent, you'll likely want to filter based on time so you can run the feedback pipeline on a schedul. The query below filters for runs since midnight, last-night UTC. You can also filter for other things, like runs without errors, runs with specific tags, etc. For more information on doing so, check out the Run Filtering guide to learn more.

midnight = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

runs = list(client.list_runs(
    project_name=project_name,
    execution_order=1,
    start_time=midnight,
    errors=False
))
```

Once you've decided the runs you want to evaluate, it's time to define the feedback pipeline.

2. Define Feedback Algorithm
All feedback needs a key and should have a (nullable) numeric score. You can apply any algorithm to generate these scores, but you'll want to choose the one that makes the most sense for your use case.

The following examples run on the "input" and "output" keys of runs. If your runs use different keys, you may have to update.

### Example A: Simple Text Statistics
We will start out by adding some simple text statistics on the input text as feedback. We use this to illustrate that you can use any simple or custom algorithm to generate scores.

Scores can be null, boolean, integesr, or float values.

Note: We are measuring the 'input' key in this example, which is used by LangChain's AgentExecutor class. You will want to confirm the key(s) you want to measure within the run's inputs or outputs dictionaries when applying this example. Common run types (like 'chat' runs) have nested dictionaries.

```python
import textstat
from langsmith.schemas import Run, Example
from langchain.schema.runnable import RunnableLambda

def compute_stats(run: Run) -> None:
    # Note: your chain's runs may have different keys.
    # Be sure to select the right field(s) to measure!
    if "input" not in run.inputs:
        return
    if run.feedback_stats and "smog_index" in run.feedback_stats:
        # If we are running this pipeline multiple times
        return
    text = run.inputs["input"]
    try:
        fns = [
            "flesch_reading_ease", 
            "flesch_kincaid_grade",
            "smog_index", 
            "coleman_liau_index", 
            "automated_readability_index",
        ]
        metrics ={
            fn: getattr(textstat, fn)(text) for fn in fns
            for fn in fns
        }
        for key, value in metrics.items():
            client.create_feedback(
                run.id,
                key=key,
                score=value, # The numeric score is used in the monitoring charts
                feedback_source_type="model",
            )
    except:
        pass
# Concurrently log feedback. You could also run this in a 'for' loop
# And not use any langchain code
_ = RunnableLambda(compute_stats).batch(
    runs, 
    {"max_concurrency": 10},
    return_errors=True,
)
```

### Example B: AI-assisted feedback
Text statistics are simple to generate but often not very informative. Let's make an example that scores each run's input using an LLM. This method lets you score runs based on targeted axes relevant to your application. You could apply this technique to select metrics as proxies for quality or to help curate data to fine-tune an LLM.

In the example below, we will instruct an LLM to score user input queries along a number of simple axes. We will be using this prompt to drive the chain.

```python
from langchain import hub

prompt = hub.pull("wfh/automated-feedback-example", api_url="https://api.hub.langchain.com")
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import collect_runs
from langchain.output_parsers.openai_functions import  JsonOutputFunctionsParser

chain = (
    prompt
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=1).bind(
        functions=[{
              "name": "submit_scores",
              "description": "Submit the graded scores for a user question and bot response.",
              "parameters": {
                "type": "object",
                "properties": {
                  "relevance": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Score indicating the relevance of the question to LangChain/LangSmith."
                  },
                  "difficulty": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Score indicating the complexity or difficulty of the question."
                  },
                  "verbosity": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Score indicating how verbose the question is."
                  },
                  "specificity": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Score indicating how specific the question is."
                  }
                },
                "required": ["relevance", "difficulty", "verbosity", "specificity"]
              }
            }
        ]
    )
    | JsonOutputFunctionsParser()
)

def evaluate_run(run: Run) -> None:
    try:
        # Note: your chain's runs may have different keys.
        # Be sure to select the right field(s) to measure!
        if "input" not in run.inputs or not run.outputs or 'output' not in run.outputs:
            return
        if run.feedback_stats and 'specificity' in run.feedback_stats:
            # We have already scored this run
            # (if you're running this pipeline multiple times)
            return
        with collect_runs() as cb:
            result = chain.invoke(
                {"question": run.inputs["input"][:3000], # lazy truncation
                 "prediction": run.outputs["output"][:3000]},
            )
            for feedback_key, value in result.items():
                score = int(value) / 5
                client.create_feedback(
                    run.id,
                    key=feedback_key,
                    score=score,
                    source_run_id=cb.traced_runs[0].id,
                    feedback_source_type="model",
                )
    except Exception as e:
        pass

wrapped_function = RunnableLambda(evaluate_run)
# Concurrently log feedback
_ = wrapped_function.batch(runs, {"max_concurrency": 10}, return_errors=True)
# Updating the aggregate stats is async, but after some time, the logged feedback stats
client.read_project(project_name=project_name).feedback_stats
```

### Example C: LangChain Evaluators
LangChain has a number of reference-free evaluators you can use off-the-shelf or configure to your needs. You can apply these directly to your runs to log the evaluation results as feedback. For more information on available LangChain evaluators, check out the open source documentation.

Below, we will demonstrate this by using the criteria evaluator, which instructs an LLM to check that the prediction against the described criteria. The criterion we specify will be "completeness".

```python
from typing import Optional
from langchain import evaluation, callbacks
from langsmith import evaluation as ls_evaluation

class CompletenessEvaluator(ls_evaluation.RunEvaluator):
    
    def __init__(self):
        criteria_description=(
            "Does the answer provide sufficient and complete information"
            "to fully address all aspects of the question (Y)?"
            " Or does it lack important details (N)?"
        )
        self.evaluator = evaluation.load_evaluator("criteria", 
                                      criteria={
                                          "completeness": criteria_description
                                      })
    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> ls_evaluation.EvaluationResult:
        if not run.inputs or not run.inputs.get('input') or not run.outputs or not run.outputs.get("output"):
            return ls_evaluation.EvaluationResult(key="completeness", score=None)
        question = run.inputs['input']
        prediction = run.outputs['output']
        with callbacks.collect_runs() as cb:
            result = self.evaluator.evaluate_strings(input=question, prediction=prediction)
            run_id = cb.traced_runs[0].id
        return ls_evaluation.EvaluationResult(
            key="completeness", evaluator_info={"__run": {"run_id": run_id}}, **result)
Here, we are using the collect_runs callback handler to easily fetch the run ID from the evaluation run. By adding it to the evaluator_info, the feedback will retain a link from the evaluated run to the source run so you can see why the tag was generated. Below, we will log feedback to all the traces in our project.

evaluator = CompletenessEvaluator()

# We can run as a regular for loop
# for run in runs:
#     client.evaluate_run(run, evaluator)

# Or concurrently log feedback
wrapped_function = RunnableLambda(lambda run: client.evaluate_run(run, evaluator))
_ = wrapped_function.batch(runs, {"max_concurrency": 10}, return_errors=True)
```

## Real-time Automated Feedback
if the metrics reveal issues, you can isolate problematic runs for debugging or fine-tuning.

Steps:
1. Define Feedback Logic: Create a RunEvaluator to calculate the feedback metrics. We will use LangChain's "CriteriaEvaluator" as an example.
2. Include in callbacks: Using the EvaluatorCallbackHandler, we can make sure the evaluators are applied any time a trace is completed.

```python
%pip install -U langchain openai --quiet
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" # Update with your API URL if using a hosted instance of Langsmith.
os.environ["LANGCHAIN_API_KEY"] = "YOUR API KEY" # Update with your API key
os.environ["LANGCHAIN_PROJECT"] = "YOUR PROJECT NAME" # Change to your project name
```

1. Define feedback logic
All feedback needs a key and should have a (nullable) numeric score. You can apply any algorithm to generate these scores, but you'll want to choose the one that makes the most sense for your use case.

The following example selects the "input" and "output" keys of the trace and returns 1 if the algorithm believes the response to be "helpful", 0 otherwise.

LangChain has a number of reference-free evaluators you can use off-the-shelf or configure to your needs. You can apply these directly to your runs to log the evaluation results as feedback.

```python
from typing import Optional
from langchain.evaluation import load_evaluator
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example

class HelpfulnessEvaluator(RunEvaluator):
    def __init__(self):
        self.evaluator = load_evaluator("score_string", criteria="helpfulness", normalize_by=10)
        
    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        if not run.inputs or not run.inputs.get('input') or not run.outputs or not run.outputs.get("output"):
            return EvaluationResult(key="helpfulness", score=None)
        result = self.evaluator.evaluate_strings(input=run.inputs['input'], prediction=run.outputs['output'])
        return EvaluationResult(**{
            "key": "helpfulness", 
            "comment": result.get("reasoning"), 
            **result})
```

Here, we are using the collect_runs callback handler to easily fetch the run ID from the evaluation run. By adding it to the evaluator_info, the feedback will retain a link from the evaluated run to the source run so you can see why the tag was generated. Below, we will log feedback to all the traces in our project.

2. Include in callbacks
We can use the EvaluatorCallbackHandler to automatically call the evaluator in a separate thread any time a trace is complete.

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("user", "{input}")])
    | ChatOpenAI()
    | StrOutputParser()
)
from langchain.callbacks.tracers.evaluation import EvaluatorCallbackHandler

evaluator = HelpfulnessEvaluator()

feedback_callback = EvaluatorCallbackHandler(
    evaluators=[evaluator]
)
queries = [
    "Where is Antioch?",
    "What was the US's inflation rate in 2018?",
    "Who were the stars in the show Friends?",
    "How much wood could a woodchuck chuck if a woodchuck could chuck wood?",
    "Why is the sky blue?",
    "When is Rosh hashanah in 2023?",
]

for query in queries:
    chain.invoke({"input": query}, {"callbacks": [feedback_callback]})
# Check out the target project to see the feedback appear as the runs are evaluated.
```

---

