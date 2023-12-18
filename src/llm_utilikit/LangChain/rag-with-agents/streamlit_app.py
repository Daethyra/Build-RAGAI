from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

# Specify the History Type
msgs = StreamlitChatMessageHistory(key="Test_App_Message_History")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")
    
template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

# Add the memory to an LLMChain as usual
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt, memory=memory)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)