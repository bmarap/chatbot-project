import os
import pandas as pd
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from models.gemini_model import get_rag_chain, get_vector_store
from datasets import Dataset

# Load env variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# 1. Setup Models for Ragas (Judges)
# We use Gemini as the judge LLM
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=api_key, 
    temperature=0
)

# We use local embeddings for evaluation metrics that need it
judge_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Wrap for Ragas
ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

# 2. Define Ground Truth Dataset
# Adjusted to 2 questions to fit within ~10 request limit
test_data = [
    {
        "question": "How do I install OpenWebUI using Docker?",
        "ground_truth": "You can install OpenWebUI using Docker by running the command: docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main"
    },
    {
        "question": "What is the default port for OpenWebUI?",
        "ground_truth": "The default port for OpenWebUI is 8080 inside the container, which is often mapped to port 3000 on the host."
    }
]

# 3. Generate Answers
print("Loading vector store...")
vectorstore = get_vector_store()
if not vectorstore:
    raise ValueError("Vector store not found! Please run the app and refresh knowledge base first.")

rag_chain = get_rag_chain(vectorstore, api_key)

print("Generating answers for test dataset...")
questions = []
ground_truths = []
answers = []
contexts = []

for item in test_data:
    q = item["question"]
    print(f"Processing: {q}")
    
    # Get answer
    response = rag_chain.invoke(q)
    
    # Get retrieved contexts (manually retrieving to inspect what was used)
    docs = vectorstore.similarity_search(q, k=4)
    ctx_text = [doc.page_content for doc in docs]
    
    questions.append(q)
    ground_truths.append(item["ground_truth"])
    answers.append(response)
    contexts.append(ctx_text)

# 4. Prepare Dataset for Ragas
data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data_dict)

# 5. Run Evaluation
print("Running Ragas evaluation...")
# Choosing 2 metrics to keep request count moderate
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

# 6. Output Results
print("\nEvaluation Results:")
print(results)

df = results.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("Results saved to ragas_results.csv")
