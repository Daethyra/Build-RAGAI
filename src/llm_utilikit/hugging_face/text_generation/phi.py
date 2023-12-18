"""# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="microsoft/phi-2", trust_remote_code=True)"""

#
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype="auto",
    device_map="cuda",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


def print_prime(n):
    """
    Print all primes between 1 and n
    """
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    print(primes)


inputs = tokenizer(
    '''def print_prime(n):
   """
   Print all primes between 1 and n
   """''',
    return_tensors="pt",
    return_attention_mask=False,
)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
