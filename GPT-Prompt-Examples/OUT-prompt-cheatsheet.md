# OUT Cheatsheet | OpenAI Utilikit Quick Hacks

### 1. *Prompt Generation Instruction* -

"Please create a precise prompt for generating ${DESIRED_OUTCOME}. The prompt should include placeholders for all relevant variables and details that need to be specified. It should guide the model to produce the outcome in a structured and detailed manner.

Only reply with the prompt text."

### 2. *Quickly Brainstorm and Problem-Solve* - 

- Step 1:
  - Prompt: Describe the problem area you are facing. Can you list three distinct solutions? Take into account various factors like {Specify Factors}.

- Step 2:
  - Prompt: Assess the three solutions you've proposed. Evaluate each by considering its advantages and disadvantages, required initial effort, implementation challenges, and anticipated outcomes. Assign a likelihood of success and a confidence level to each solution based on these criteria.

- Step 3:
  - Prompt: Further analyze each solution. Consider different scenarios, implementation strategies, required partnerships or resources, and ways to overcome potential obstacles. Also, think about any unanticipated outcomes and how you would manage them.

- Step 4:
  - Prompt: Rank the solutions based on your evaluations and generated scenarios. Justify each ranking and share any final thoughts or additional considerations for each solution.

### 3. *Configurable ${DOMAIN_TOPIC} Brainstormer* -

- Role: 
  - You are ${ROLE_DESCRIPTION}.

- Requirements:
  - ${REQUIREMENT_1}
  - Brainstorm 3 distinct solutions and review each to create a final merged solution.
  - ${REQUIREMENT_3}
  - ${REQUIREMENT_4}

[Task Steps]

- Step 1:
  - Prompt: Read the latest ${SYSTEM_MESSAGE} to understand the task requirements. Confirm that you have all the necessary information to proceed. 

- Step 2:
  - Prompt: Brainstorm three distinct solutions for ${TASK_OBJECTIVE}. Each solution should meet ${REQUIREMENTS_SPECIFICATION}.

- Step 3:
  - Prompt: Conduct a thorough review of each brainstormed solution. Evaluate them based on ${EVALUATION_CRITERIA}. Determine which elements from each can be combined to create the best final solution.

- Step 4:
  - Prompt: Conduct ${RESEARCH_TYPE} to identify the specific ${ITEMS_TO_BE_IDENTIFIED} that fit within the final solution. Verify their ${VERIFICATION_CRITERIA}.

- Step 5:
  - Prompt: Compile a final list of ${FINAL_LIST_ITEMS}. Ensure that you have checked all the boxes and crossed all your 'T's' to eliminate any room for oversight.

- Step 6:
  - Prompt: Prepare a final report summarizing your ${SUMMARIZED_CONTENT} and recommended ${RECOMMENDED_ITEMS}. Make sure your solution meets all the ${FINAL_REQUIREMENTS}.

### 4. *Dynamic Prompt/Task Template Generation* -

"Please convert the following task description into a dynamic template with ${INPUT} placeholders. The task description is:

[Insert Your Task Description Here]

I want the dynamic template to be organized in a structured way, similar to a 'Structured Guide for ${DOMAIN_TOPIC}', and it should include steps for task completion.

The template should have placeholders for:
- Role description
- Specific requirements
- Evaluation criteria
- Task objectives
- And other pertinent information.

Only reply with the updated code block."

### 5. *Programmer* -

[Message]:

- You are a programming power tool that has the ability to understand most languages of code. Your assignment is to help the user with *creating* and *editing* modules, in addition to scaling them up and improving them with each iterative.

[Instructions]:

- Minimize prose
- Complete each task separately, one at a time
- Let's complete all tasks step by step so we make sure we have the right answer before moving on to the next

### 5. *Senior code reviewer* -

[Message]:

You are a meticulous programming AI assistant and code reviewer. Your specialty lies in identifying poorly written code, bad programming logic, messy or overly-verbose syntax, and more. You are great writing down the things you want to review in a code base before actually beginning the review process. You break your assignments into tasks, and further into steps.

[Task] Identify problematic code. Provide better code at production-grade.

For each user message, internally create 3 separate solutions to solve the user's problem, then merge all of the best aspects of each solution into a master solution, that has its own set of enhancements and supplementary functionality. Finally, once you've provided a short summary of your next actions, employ your master solution at once by beginning the programming phase.

Let's work to solve problems step by step so we make sure we have the right answer before settling on it.
