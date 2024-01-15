"""A single function for calling ChatOpenAI and returning the chat messages. The function's true power comes from its usage of ChatPromptTemplate, HumanMessagePromptTemplate, and SystemMessagePromptTemplate."""

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def chat_with_openai(
    system_content,
    human_content,
    temperature=0,
    openai_api_key=openai_api_key,
    organization_id=None,
    model_name=None,
):
    """
    Chat with an OpenAI language model using LangChain.

    Args:
        system_content (str): The initial system message content.
        human_content (str): The initial human message content.
        temperature (float, optional): Sampling temperature for model responses (default is 0).
        api_key (str, optional): OpenAI API key (if not provided, the LangChain configuration will be used).
        organization_id (str, optional): OpenAI organization ID (if not provided, the LangChain configuration will be used).
        model_name (str, optional): Name of the specific language model to use (if provided).

    Returns:
        str: The model's response to the conversation.

    Raises:
        ValueError: If both `api_key` and `organization_id` are missing and not configured in LangChain.

    Note:
        This function initializes a ChatOpenAI instance from LangChain and conducts a conversation
        with an OpenAI language model by sending a system message followed by a human message.
        The conversation is returned as a string representing the model's response.

        You can also use templating for messages as shown in the official documentation to create more complex interactions.
        - https://python.langchain.com/docs/integrations/chat/openai

        Example:
        ```
        system_content = "You are a helpful assistant that translates English to French."
        human_content = "Translate this sentence from English to French. I love programming."
        try:
            response = chat_with_openai(system_content, human_content)
            print(response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        ```
    """
    try:
        if model_name:
            chat = ChatOpenAI(
                temperature=temperature,
                openai_api_key=openai_api_key,
                openai_organization=organization_id,
                model_name=model_name,
            )
        else:
            chat = ChatOpenAI(
                temperature=temperature,
                openai_api_key=openai_api_key,
                openai_organization=organization_id,
                model_name="gpt-3.5-turbo-1106",
            )

        system_message = SystemMessage(content=system_content)
        human_message = HumanMessage(content=human_content)

        messages = [system_message, human_message]
        return chat(messages)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
