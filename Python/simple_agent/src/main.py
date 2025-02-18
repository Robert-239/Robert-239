import os


from tavily import TavilyClient, AsyncTavilyClient
from pydantic_ai import Agent
from dotenv import load_dotenv

import json

from tavily.errors import UsageLimitExceededError


#top line code
load_dotenv()
#
print(os.environ["TAVILY_API_KEY"])

tavily_client = AsyncTavilyClient(api_key= os.environ["TAVILY_API_KEY"])

try:
    #try a search
except UsageLimitExceededError:
    print(UsageLimitExceededError)
