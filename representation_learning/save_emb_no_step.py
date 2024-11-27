from transformers import BertModel, BertTokenizer, BertConfig
import torch
import json
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize


# Get the folder to load the trained model
folder_saved_model = '/home/xiongzj/myProjects/KCQRL-main/models_CL-cluster_only-original-KC' # Experiment folder
embeddings_name = "qid2content_CL-cluster_only-original-KC.json"
path_data_questions = '/home/xiongzj/myProjects/KCQRL-main/data/moocradar-C_746997/metadata/questions_translated_kc_sol_annotated_mapped_original_kc.json'
path_kc_questions_map = '/home/xiongzj/myProjects/KCQRL-main/data/moocradar-C_746997/metadata/kc_questions_map_original_kc.json'
embeddings_save_folder = "/home/xiongzj/myProjects/KCQRL-main/representation_learning"

with open(path_data_questions, 'r') as file:
    data_questions = json.load(file)

with open(path_kc_questions_map, 'r') as file:
    kc_questions_map = json.load(file)

if not os.path.exists(embeddings_save_folder):
    os.makedirs(embeddings_save_folder)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(folder_saved_model + '/tokenizer')

# Create a configuration object or load it if you have saved one
config = BertConfig.from_pretrained('bert-base-uncased')

# Initialize the model with this configuration
model = BertModel(config)

# Adjust the model's token embeddings to account for new tokens before loading the weights
model.resize_token_embeddings(len(tokenizer))

# Load the model weights
model.load_state_dict(torch.load(folder_saved_model + '/bert_finetuned.bin'))

# Move the model to the appropriate computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to training or evaluation mode as needed
model = model.eval()  # or model.train() if you continue training

# Constants
BATCH_SIZE = 1024  # Define your batch size here


# Helper function to batch text data and convert to embeddings
def text_to_embeddings(texts, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :])  # Extract [CLS] token embeddings
    return torch.cat(embeddings, dim=0)

list_questions = [value['question'] for key, value in data_questions.items()]

# Prepend special tokens
questions = ['[Q] ' + q for q in list_questions]

# Get the embeddings
question_embeddings = text_to_embeddings(questions)

# Convert these embeddings to numpy array or lists to have necessary pre-computations
np_question_embeddings = question_embeddings.cpu().detach().numpy()

dict_emb = {}
for i in range(len(np_question_embeddings)):
    emb = np_question_embeddings[i].copy().reshape(1,-1)
    norm_emb = normalize(emb, axis=1, norm='l2').flatten()
    dict_emb[str(i)] = norm_emb.tolist()

save_path = os.path.join(embeddings_save_folder, embeddings_name)

with open(save_path, 'w') as f:
    json.dump(dict_emb, f)
