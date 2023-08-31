
import csv, os
from dotenv import find_dotenv, load_dotenv
import pinecone
import openai
import tiktoken

load_dotenv(find_dotenv())

PINDIR = "." # "/home/skanduru/JobSearch/AutoRecAI/Design/ml-25m"
pinecone.init(api_key=os.environ['PINECONE_API_KEY'])
files = "movies.csv"
emb_model = "text-embedding-ada-002"


def get_vector(id: str, model: str):
    """
    encoding = tiktoken.encoding_for_model(emb_model)
    return encoding.encode(id)
    """
    response = openai.Embedding.create(model = emb_model, input = id)
    return response.data[0].embedding

def create_index(file_path, index_name):

    # Read the CSV file and upsert the data into the Pinecone index
    with open(PINDIR + "/" + file_path) as f:
        reader = csv.reader(f)
        row0 = None

        for row in reader:
            row0 = row
            break

        # pinecone.create_index(name=index_name, dimension=1536, metadata_config = metadata_config, metric= "cosine")
        pinecone.create_index(name=index_name, dimension=1536, metric= "cosine")

        index = pinecone.Index(index_name=index_name)
        entries = []

        try:
         for row in reader:
            # Extract the data from the row
            id = '[CLS] ' + ' [SEP] '.join(row) + ' [SEP]'
            metadata = {}
            for i in range(len(row)):
                metadata[row0[i]] = row[i]

            vector = get_vector(id, emb_model)
            # vector = response["embeddings"]
            # vector = [float(x) for x in row[1:]]
            # metadata = {"title": row[1], "genres": row[2]}
            # Create an entry as a dictionary with id and vector keys
            if len(entries) < 16:
                # entry = {"id": id, "values": vector}
                entry = (id, vector, metadata)
                entries.append(entry)
                index.upsert(vectors=entries)
                entries = []
            else:

                # Upsert the data into the Pinecone index
                # pinecone.upsert(index_name=index_name, ids=[id], vectors=[vector], metadata=[metadata])
                pinecone.upsert(entries)
                entries = []
         if entries:
            pinecone.upsert(entries)
        except:
            pass

def get_movie_recommendations(genre, number):
    index = pinecone.Index(index_name='movies-index')
    query_vec = get_vector(f'[CLS] {genre} [SEP]', emb_model)
    search = index.query(top_k=number, vector=query_vec, include_metadata = True)
    return search

