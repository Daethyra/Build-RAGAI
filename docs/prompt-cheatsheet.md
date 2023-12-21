# Prompt Cheatsheet

- '${MASK}' is a placeholder for a masked word or phrase that *YOU* need to fill in.

## 1. *Instruction: Generate Prompt


Please create a precise prompt for generating ${DESIRED_OUTCOME}. The prompt should include placeholders for all relevant variables and details that need to be specified. It should guide the model to produce the outcome in a structured and detailed manner.

Only reply with the prompt text.


## 2. *Masked Language Model Mimicry Prompt*

AI Chatbot, your task is to mimic how fill-mask language models fill in masked words or phrases. When I provide you with a sentence that contains one or more masked positions, denoted by ${MASK}, please replace the ${MASK} with the most appropriate word or phrase based on the surrounding context.

For example, if I say, "The ${MASK} jumped over the moon", you might respond with "The cow jumped over the moon".

Input Sentence: ${INPUT_SENTENCE_WITH_MASK}
Context (if any): ${ADDITIONAL_CONTEXT}

Please output the sentence with all masked positions filled in a manner that is coherent and contextually appropriate. Make sure to include the filled mask(s) in your response.

Output Format: [Original Sentence]: [Filled Sentence]


## 3. *Quickly Brainstorm and Problem-Solve* 

- Step 1:
  - Prompt: Describe the problem area you are facing. Can you list three distinct solutions? Take into account various factors like {Specify Factors}.

- Step 2:
  - Prompt: Assess the three solutions you've proposed. Evaluate each by considering its advantages and disadvantages, required initial effort, implementation challenges, and anticipated outcomes. Assign a likelihood of success and a confidence level to each solution based on these criteria.

- Step 3:
  - Prompt: Further analyze each solution. Consider different scenarios, implementation strategies, required partnerships or resources, and ways to overcome potential obstacles. Also, think about any unanticipated outcomes and how you would manage them.

- Step 4:
  - Prompt: Rank the solutions based on your evaluations and generated scenarios. Justify each ranking and share any final thoughts or additional considerations for each solution.


## 4. *Configurable ${DOMAIN_TOPIC} Brainstormer* 

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


## 5. *Guide-Creation Template for AI Assistant's Support* 

Request: Create a comprehensive and structured guide to assist users in understanding and utilizing *[Specific Tool or Library]*. This guide should be designed to provide clear, actionable information and support users in their projects involving *[Specific Use Case or Application]*.

Purpose: To offer users a detailed and accessible resource for *[Specific Tool or Library]*, enhancing their ability to effectively employ it in their projects.

Requirements for the Guide:

- Project Overview: Provide a general introduction to *[Specific Tool or Library]*, including its primary functions and relevance to *[Specific Use Case or Application]*.
- Key Features and Tools: Describe the essential features and tools of *[Specific Tool or Library]*, highlighting how they can be leveraged in practical scenarios.
- User Instructions: Offer step-by-step guidance on how to set up and utilize *[Specific Tool or Library]*, ensuring clarity and ease of understanding for users of varying skill levels.
- Practical Examples: Include examples that demonstrate the application of *[Specific Tool or Library]* in real-world scenarios, relevant to *[Specific Use Case or Application]*.
- Troubleshooting and Support: Provide tips for troubleshooting common issues and guidance on where to seek further assistance or resources.
- Additional Resources: List additional resources such as official documentation, community forums, or tutorials that can provide further insight and support.

Goal: To create a user-friendly, informative guide that empowers users to effectively utilize *[Specific Tool or Library]* for their specific needs and projects, thereby enhancing their skills and project outcomes.

For each user request, brainstorm multiple solutions or approaches, evaluate their merits, and synthesize the best elements into a comprehensive response. Begin implementing this approach immediately to provide the most effective assistance possible.

