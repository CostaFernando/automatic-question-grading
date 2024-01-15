import pandas as pd
import numpy as np
from openai import OpenAI

open_ai_client = OpenAI(api_key="OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return open_ai_client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(embedding1, embedding2):
  return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

student_answers_df = pd.read_csv('Redes de Computadores - answers.csv')
questions_df = pd.read_csv('Redes de Computadores - questions.csv')

pivot_df = pd.merge(questions_df, student_answers_df, on='Question Number')
pivot_df = pivot_df[['Index', 'Question Number', 'Question', 'Correct Answer', 'Answer']]
pivot_df = pivot_df.sort_values(by=['Index'])

pivot_df['Correct Answer Embedding'] = pivot_df['Correct Answer'].apply(get_embedding)
pivot_df['Answer Embedding'] = pivot_df['Answer'].apply(get_embedding)

pivot_df['Similarity'] = pivot_df.apply(lambda row: cosine_similarity(row['Correct Answer Embedding'], row['Answer Embedding']), axis=1)

pivot_df[['Question Number', 'Similarity']].to_csv('Cosine Similarity.csv', index=False)