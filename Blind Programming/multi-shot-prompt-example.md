## The code below is exactly what was sent to ChatGPT 
Specifications: `(Model=4, Plugins=['webpilot', 'metaphor'])`, 
starting with `<!document-begin>` at 9:33PM on 7/5/23.
- Please view the [raw version]()

---
---
<!document-begin>
```
## [System message(s)]:
    - "You are an AI programming assistant that is skilled in brainstorming different deployment ideas for new projects, and are also an expert in coding in many different languages to create any application, with dedication and diligence. You are so smart that you can access the internet for resources, references, and documentation when you're stuck, or aren't sure if the code you're writing is syntactically correct. You're very good at double checking your work to ensure you have the right answer before moving on, or sharing your findings."

### [User message(s)]:
    - "I need to make a simple Swift application that counts boxes and labels them based on the label on the box, and how it looks. I intend to use a GPT model, either 3.5-turbo[...] OR GPT-4[...]"
---
[User message(s)]:
    - "Please see the code examples below, and ensure you remember we're going to build a Swift application so you need to think and plan ahead as you learn more and more about the task at hand and what we'll need to accomplish our idea application."
    - "We need to have a simple login interface that then leads to a homepage where you can create, edit, and delete groups that will hold subgroups that will be named based on what they're counting. Let's say, for example, that we have the top-level grouping named 'Freezer', and then a subgrouping of 'Macaroon Boxes', another of 'Bake-Offs', and finally one of 'Macaroons'. In this example we're counting different objects with our camera to save time for a small local business."
    - "Let's just focus on the Swift programming aspects for now, we'll program components that are based on other languages, like Python3, sometime in the future once the Swift app is fully functional w/o the required remote database operations and more."


---

[Task 1]:
"Brainstorm 3 separate solutions that will fulfill the user's requirements for their application. You will need to consider a variety of factors and variables, even those not immediately apparent. For example, you need to consider what end-goal state the code modules should be in. As in, what modules are required? How will the back-end database be handled, or what about the API calls to OpenAI's endpoints? How will the modules be resilient and have try, retry, and break conditions?"

[Task 2]:
"""
- Step 1 -
"Review all 3 solutions and extract all of the best ideas from each module and plan to implement them in a final solution, in your head."
- Step 2 -
"Then extract all of the weak points, flaws, and oversight in each module, and other potential flaws that may arise from new programming decisions, to program measures that account for those shortcomings before they ever arise during testing. Spend very much time on this task."
- Step 3 -
"Finalize your master solution that meets all of the requirements found in this task's first two steps."
"""

Ensure that you always utilize structured data where optimal for lightning fast calls during runtime.

---

[Supplementary information, data, and documentation]:
- https://developer.apple.com/documentation/swift
- https://openai.com/customer-stories/be-my-eyes
- https://openai.com/blog/function-calling-and-other-api-updates
- https://gptstore.ai/plugins/webpilotai-com(recommended reading)
-- Seems like you could link images to GPT, and be very clear about viewing the image with a plugin, and then respond to the user based on what they need relative to the media. If you implement this idea, you will need to use extremely clear and concise action steps for the bot to take so that it does everything we intend and need for it do it rather than having any type of variance. Essentially, we're going for a temperature level of 0.
- https://platform.openai.com/docs/api-reference/chat(recommended reading)
- https://platform.openai.com/docs/guides/gpt
- https://platform.openai.com/docs/models
- https://github.com/microsoft/TaskMatrix
```
<!document-end>

-----


[System message(s)]:
"Please read the entire command sheet you just received before doing anything. Ensure you have a complete understanding of the entire assignment sheet and then tell me when you're ready to begin exploring the links provided. Then, you'll need to tell me when you're ready to begin the next part, which is where we will actually begin working on the tasks, and their steps, one by one. So let's do things 'step by step' so we make sure we have the right answer before moving on to the next one."
