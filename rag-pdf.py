#import fitz
import os
import re
import requests

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from chromadb.api.types import EmbeddingFunction
from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Optional, Any

def pdf_to_text(path: str,
                start_page: int = 1,
                end_page: Optional[int | None] = None) -> list[str]:
    """
    Converts PDF to plain text.

    Params:
        path (str): Path to the PDF file.
        start_page (int): Page to start getting text from.
        end_page (int): Last page to get text from.
    """
    loader = PyPDFLoader("pdfs/paper_flowers.pdf")
    pages = loader.load()
    total_pages = len(pages)

    if end_page is None:
        end_page = len(pages)

    text_list = []
    for i in range(start_page-1, end_page):
        text = pages[i].page_content
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text_list.append(text)

    return text_list

# PDF files available:
#    "pdfs/pie_recipe.pdf"
#    "pdfs/paper_flowers.pdf"
text_list = pdf_to_text("pdfs/paper_flowers.pdf")
print(text_list)


def text_to_chunks(texts: list[str],
                   word_length: int = 150,
                   start_page: int = 1) -> list[list[str]]:
    """
    Splits the text into equally distributed chunks.

    Args:
        texts (str): List of texts to be converted into chunks.
        word_length (int): Maximum number of words in each chunk.
        start_page (int): Starting page number for the chunks.
    """
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i + word_length]
            if (i + word_length) > len(words) and (len(chunk) < word_length) and (
                    len(text_toks) != (idx + 1)):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx + start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks
chunks = text_to_chunks(text_list)

for chunk in chunks:
    print(chunk + '\n')

%%time
# Load the model from TF Hub
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
emb_function = MiniLML6V2EmbeddingFunction()

def get_text_embedding(texts: list[list[str]],
                       batch: int = 1000) -> list[Any]:
    """
    Get the embeddings from the text.

    Args:
        texts (list(str)): List of chucks of text.
        batch (int): Batch size.
    """
    embeddings = []
    for i in range(0, len(texts), batch):
        text_batch = texts[i:(i + batch)]
        # Embeddings model
        emb_batch = emb_function(text_batch)
        embeddings.append(emb_batch)
    embeddings = np.vstack(embeddings)
    return embeddings

embeddings = get_text_embedding(chunks)

print(embeddings.shape)
print(f"Our text was embedded into {embeddings.shape[1]} dimensions")

print(embeddings[0])

# question = 'How many people does this pie serve?'
question = 'What are edible flowers?'
emb_question = emb_function([question])

# Create a t-SNE model
tsne = TSNE(n_components=2, random_state=42)
embeddings_with_question = np.vstack([embeddings, emb_question])
embeddings_2d = tsne.fit_transform(embeddings_with_question)

embeddings_2d.shape


def visualize_embeddings(embeddings_2d: np.ndarray,
                         question: Optional[bool] = False,
                         neighbors: Optional[np.ndarray] = None) -> None:
    """
    Visualize 384-dimensional embeddings in 2D using t-SNE, label each data point with its index,
    and optionally plot a question data point as a red dot with the label 'q'.

    Args:
        embeddings (numpy.array): An array of shape (num_samples, 384) containing the embeddings.
        question (numpy.array, optional): An additional 384-dimensional embedding for the question.
                                          Default is None.
    """

    # Scatter plot the 2D embeddings and label each data point with its index
    plt.figure(figsize=(10, 8))
    num_samples = embeddings.shape[0]
    if neighbors is not None:
        for i, (x, y) in enumerate(embeddings_2d[:num_samples]):
            if i in neighbors:
                plt.scatter(x, y, color='purple', alpha=0.7)
                plt.annotate(str(i), xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')
            else:
                plt.scatter(x, y, color='blue', alpha=0.7)
                plt.annotate(str(i), xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')
    else:
        for i, (x, y) in enumerate(embeddings_2d[:num_samples]):
            plt.scatter(x, y, color='blue', alpha=0.7)
            plt.annotate(str(i), xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')

    # Plot the question data point if provided
    if question:
        x, y = embeddings_2d[-1]  # Last point corresponds to the question
        plt.scatter(x, y, color='red', label='q')
        plt.annotate('q', xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')

    plt.title('t-SNE Visualization of 384-dimensional Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

visualize_embeddings(embeddings_2d[:-1])
visualize_embeddings(embeddings_2d, True)

nn_2d = NearestNeighbors(n_neighbors=5)
nn_2d.fit(embeddings_2d[:-1])

neighbors = nn_2d.kneighbors(embeddings_2d[-1].reshape(1, -1), return_distance=False)
neighbors
visualize_embeddings(embeddings_2d, True, neighbors)
nn = NearestNeighbors(n_neighbors=5)
nn.fit(embeddings)

# question = 'How many people does this pie serve?'
question = 'What are edible flowers?'
emb_question = emb_function([question])

neighbors = nn.kneighbors(emb_question, return_distance=False)
neighbors

topn_chunks = [chunks[i] for i in neighbors.tolist()[0]]


def build_prompt(question):
    prompt = ""
    prompt += 'Search results:\n'

    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "Cite each reference using [Page Number] notation (every result has this number at the beginning). " \
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects " \
              "with the same name, create separate answers for each. Only include information found in the results and " \
              "don't add any additional information. Make sure the answer is correct and don't output false content. " \
              "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier " \
              "search results which has nothing to do with the question. Only answer what is asked. The " \
              "answer should be short and concise."

    prompt += f"\n\n\nQuery: {question}\n\nAnswer: "

    return prompt

prompt = build_prompt(question)
print(prompt)

# Config watsonx.ai environment
load_dotenv()
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }


def send_to_watsonxai(prompts,
                      model_name="google/flan-ul2",
                      decoding_method="greedy",
                      max_new_tokens=100,
                      min_new_tokens=30,
                      temperature=1.0,
                      repetition_penalty=2.0
                      ):
    '''
   helper function for sending prompts and params to Watsonx.ai

    Args:
        prompts:list list of text prompts
        decoding:str Watsonx.ai parameter "sample" or "greedy"
        max_new_tok:int Watsonx.ai parameter for max new tokens/response returned
        temperature:float Watsonx.ai parameter for temperature (range 0>2)
        repetition_penalty:float Watsonx.ai parameter for repetition penalty (range 1.0 to 2.0)

    Returns: None
        prints response
    '''

    # Instantiate parameters for text generation
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
    }

    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id)

    for prompt in prompts:
        print(model.generate_text(prompt))

# Example questions for the paper:
#   Summarize the irrigation and biotechnology of this paper
#   What are edible flowers?
#   Summarize the conclusion of this study:
#   What is the main conclusion of the study?

# Example questions for the recipe:
#   Summarize this recipe process
#   How many people does this pie serve?
#   How many eggs are necessary to make this pie?

question = "Summarize the irrigation and biotechnology of this paper"
prompt = build_prompt(question)

send_to_watsonxai(prompts=[prompt], min_new_tokens=1)