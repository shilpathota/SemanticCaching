import getpass
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def load_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("> OpenAI API key is already loaded in the environment")


config = dict(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    cache_name=os.getenv("CACHE_NAME", "semantic-cache"),
    distance_threshold=float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.3")),
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
)
