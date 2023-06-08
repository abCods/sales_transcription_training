import os
from flask import Flask, request, jsonify
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from config import settings
import openai

app = Flask(__name__)

openai.api_key = settings.get("openai.api_key");

# Set OPENAI_API_KEY if not already set
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.get("openai.api_key");

documents = SimpleDirectoryReader("data").load_data()

# Try load index from Storage
try:
    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    # Load Index
    index = load_index_from_storage(storage_context=storage_context)

# Otherwise, create index from documents
except:
    index = GPTVectorStoreIndex.from_documents(documents=documents)
    index.storage_context.persist()


query_engine = index.as_query_engine()

@app.route('/api/query', methods=['POST'])
def process_query():
    prompt = request.json['prompt']
    response = query_engine.query(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
