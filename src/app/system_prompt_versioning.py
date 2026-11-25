from langfuse import get_client
from dotenv import load_dotenv

# load the api keys
load_dotenv()

chunk_size = 300
chunk_overlap = 30
output_dimensions = 1024
# k = 3
k=5

system_prompt = """You are a helpful assistant. Answer the user query
                    based on the given context only.  If you do not know the answer say I don't know. Do not add any preamble to the response. Respond back in simple language and keep a persona of a mentor that is solving doubts of his students. Your answer should have a personal feeling to them."""


# add system prompt to langfuse

langfuse = get_client()

# comment code for update checking
"""
created_prompt = langfuse.create_prompt(
    name="rag_app_system_prompt",
    type = "text",
    prompt=system_prompt,
    # did change to check in langfuse
    # labels=["baseline"],
    labels=["staging"],
    config={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "output_dims": output_dimensions,
            "k": k
    }
)
print(created_prompt.prompt)
print(created_prompt.version)
print(created_prompt.labels)
"""

# you can update label
langfuse.update_prompt(
    name = "rag_app_system_prompt",
    version=2,
    new_labels = ["production"]
)