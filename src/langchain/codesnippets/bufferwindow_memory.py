from langchain.memory import ConversationBufferWindowMemory


def window_memory(
    messages, ai_prefix="AI", human_prefix="Human", k=10, return_messages=False
):
    """
    Add conversation memory to an LLM application using LangChain's ConversationBufferWindowMemory

    Args:
        messages (List[BaseMessage]): List of conversation messages to add to memory
        ai_prefix (str): Prefix to use for AI messages in memory
        human_prefix (str): Prefix to use for human messages in memory
        k (int): Number of messages to store in memory buffer
        return_messages (bool): Whether to return messages in memory buffer as list of messages
            or as a concatenated string

    Returns:
        None
    """
    memory = langchain.memory.buffer_window.ConversationBufferWindowMemory(
        ai_prefix=ai_prefix,
        human_prefix=human_prefix,
        k=k,
        return_messages=return_messages,
    )

    """
    use the `save_context` method of the `memory` object to save the context,
    passing an empty dictionary as the first argument and the messages parameter as the second argument
    """
    # only save memory, don't return it
    memory.save_context({}, messages)
    
    # Optional: uncomment to return the memory object for further use
    # return memory