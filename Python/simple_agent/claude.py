import anthropic
from dotenv import load_dotenv
import os
load_dotenv()

claud_key = os.environ['anthropic_api_key']

print(claud_key)

client = anthropic.Anthropic(api_key= claud_key)
