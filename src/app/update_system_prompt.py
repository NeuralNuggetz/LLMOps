from distro import name
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

# shifting the labels and development
langfuse = get_client()

langfuse.update_prompt(
    name="rag_app_system_prompt",
    version=None,
    new_labels=["staging"]
)