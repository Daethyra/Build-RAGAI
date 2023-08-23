# User "Role" Prompt Examples
` '---' = END `

## Troubleshooting code

[task]:"analyze all code and the traceback error. create a multi-step plan to solve the error, enhance the code logic to prevent future errors, and add more detailed logging to the `finaid_train.py` module."
---


## *1. Iterative Processing* - 
   ! Optimal Prompt due to brevity in prose and hightens accuracy to user's requests by ~80% 
   ### [AI Explained's Video](https://www.youtube.com/watch?v=wVzuvf9D9BU)

[Instructions]:
- Complete each task separately 
- Let's complete all tasks step by step so we make sure we have the right answer before moving on to the next
---


## *2. "Tree of Thoughts"* - 
   A Short Preliminary Power Prompt

- Step1 :
    - Prompt: I have a problem related to [describe your problem area]. Could you brainstorm three distinct solutions? Please consider a variety of factors such as [Your perfect factors]
- Step 2:
    - Prompt: For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. Assign a probability of success and a confidence level to each option based on these factors
- Step 3:
    - Prompt: For each solution, deepen the thought process. Generate potential scenarios, strategies for implementation, any necessary partnerships or resources, and how potential obstacles might be overcome. Also, consider any potential unexpected outcomes and how they might be handled.
- Step 4:
    - Prompt: Based on the evaluations and scenarios, rank the solutions in order of promise. Provide a justification for each ranking and offer any final thoughts or considerations for each solution
---


## *3. Task-oriented Processing* -
   For when you need to be super specific

[Instructions]:
- Minimize prose to avoid over-tokenization
- Focus on one task at a time(iterative analysis)
- Complete each task separately 
- Let's complete all tasks step by step so we make sure we have the right answer before moving on to the next
---


## *4. Breaking down the above paragraph* -
-   Sometimes a short colloquial prompt is most powerful.

"Let's do things step by step so we make sure we have the right answer before moving on to the next one. You're to consider each sentence above to be a step. Before executing a step, ask for permission."
