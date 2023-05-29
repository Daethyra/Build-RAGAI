import os
import openai
from dotenv import load_dotenv, find_dotenv
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import logging
import panel as pn

# Setting up logging
logging.basicConfig(
    filename='app.log', 
    filemode='a', 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

pn.extension()
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

class OpenAI_Chat:
    def __init__(self, model=os.getenv('MODEL', 'gpt-3.5-turbo'), temperature=os.getenv('TEMPERATURE', 0)):
        self.model = model
        self.temperature = float(temperature)
        self.messages = []

    def get_response(self, role, message):
        self.messages.append({"role": role, "content": message})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )
        return response.choices[0].message["content"] # type: ignore

    def reset_conversation(self):
        self.messages = []
        
    def add_initial_messages(self, messages):
        self.messages.extend(messages)

class ChatApplication(tk.Tk):
    def __init__(self, chat_model, messages=None, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        if messages is None:
            messages = []
        self.title("Chatbot")
        self.configure(bg='white')
        self.chat_model = chat_model
        self.chat_model.add_initial_message(messages)

        # Role variable for checkbutton
        self.role_var = tk.StringVar()
        self.role_var.set('user')

        # Make window rounded
        self.attributes('-alpha', 0.9)  
        self['bg']='white'

        self.setup_ui()

    def setup_ui(self):
        self.geometry('800x600')  # Increase window size

        self.top_frame = tk.Frame(self, bg='white')
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.model_label = tk.Label(self.top_frame, text=f"Model: {self.chat_model.model}", bg='white', fg='black')
        self.model_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.text_area = scrolledtext.ScrolledText(self.top_frame, wrap = tk.WORD, width=40, height=10, font =("Arial", 15))
        self.text_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.bottom_frame = tk.Frame(self, bg='white')
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.message_entry = tk.Entry(self.bottom_frame, width=30, font=("Arial", 15))
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.message_entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.bottom_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.reset_button = tk.Button(self.bottom_frame, text="Reset", command=self.reset_conversation)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Role selection
        self.role_button = ttk.Checkbutton(self.bottom_frame, text="System", onvalue='system', offvalue='user', variable=self.role_var)
        self.role_button.pack(side=tk.LEFT, padx=5, pady=5)

    def send_message(self, event=None):
        message = self.message_entry.get()
        role = self.role_var.get()

        if not message.strip():
            return

        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"\n{role.capitalize()}: {message}\n")
        self.text_area.config(state=tk.DISABLED)

        self.message_entry.delete(0, tk.END)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.chat_model.get_response, role, message)
            try:
                response = future.result()
                self.text_area.config(state=tk.NORMAL)
                self.text_area.insert(tk.END, f"Bot: {response}\n")
                self.text_area.config(state=tk.DISABLED)
                logging.info(f"User: {message}, Bot: {response}")
            except Exception as e:
                    messagebox.showerror("Error", str(e))
                    logging.error(f"Error while getting response: {str(e)}")

    def reset_conversation(self):
        self.chat_model.reset_conversation()
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)
        logging.info("Conversation reset")


if __name__ == "__main__":
    Instructions = f"""

[Instructions]: \
    - You are a '20 something' cyberpunk that speaks like they're from 2023.\
    - You are skilled in programming, problem solving, and processing text.\
    - Your name is 'Aebbi'.\
    -- You need to complete assignments step by step to ensure you have the right answer.\
    - Your main job is to assist the user with whatever they're working on.\
    - Await user input for further instructions.
"""

    messages = [
        {
            "role": "system",
            "content": Instructions
        }
    ]

    chat_model = OpenAI_Chat()
    app = ChatApplication(chat_model, messages)  # pass messages as the second argument
    app.mainloop()