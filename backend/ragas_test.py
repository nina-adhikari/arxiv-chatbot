import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from os import getenv as env
import requests
from  urllib.parse import quote

from ragas import evaluate
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    faithfulness,
    context_recall,
    context_precision,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_mistralai import ChatMistralAI
# from langchain_anthropic.chat_models import ChatAnthropic
from langchain_community.document_loaders import DataFrameLoader

load_dotenv()
API_URL = env('API_URL')
FILENAME = 'test_papers.feather'

GPT = ChatOpenAI(model="gpt-3.5-turbo")
# MISTRAL = ChatMistralAI(model="open-mistral-7b", api_key = env("MISTRAL_API_KEY"))
# CLAUDE = ChatAnthropic()

TEST_SIZE = 20

METRICS = [
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    faithfulness,
    context_recall,
    context_precision,
]

def load_testset(filename):
    test_papers = pd.read_feather(filename)

    loader = DataFrameLoader(test_papers, page_content_column="text")
    docs = loader.load()

    for doc in docs:
        doc.metadata['filename'] = doc.metadata['source']
    return docs


def get_generator(llm):
    generator_llm = llm
    critic_llm = llm
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    return TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )


def generate_questions(generator, testset, test_size):
    return generator.generate_with_langchain_docs(
        testset,
        test_size=test_size,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
    )

def get_answer(query):
    query = quote(query, safe='/:')
    stream = requests.post(
        API_URL,
        params={'user_id': 0, 'message': query}
    )
    answer = stream.json()['output']
    context = stream.json()['intermediate_steps'][0][1]['context']
    sources = []
    for source in context:
         sources.append(source['page_content'])

    return [answer, sources]

def get_predictions(testset):
    test_df = testset.to_pandas()
    y_pred = test_df['question'].apply(get_answer)

    y_contexts = [pred[1] for pred in y_pred]
    y_answer = [pred[0] for pred in y_pred]

    test_df['answer'] = y_answer
    test_df['contexts'] = y_contexts

    return Dataset.from_pandas(test_df[['question', 'contexts', 'answer', 'ground_truth']])

def get_evaluation(predictions, metrics, llm):
    return evaluate(
        predictions,
        metrics,
        llm=llm
    )

def run_pipeline(test_file=FILENAME, test_size=TEST_SIZE, metrics=METRICS, generate_llm=GPT, evaluate_llm=GPT):
    generator = get_generator(generate_llm)
    testset = load_testset(test_file)
    questions = generate_questions(generator, testset, test_size)
    predictions = get_predictions(questions)
    results = get_evaluation(predictions, metrics, evaluate_llm)

    return results


if __name__ == "__main__":
    print(run_pipeline())