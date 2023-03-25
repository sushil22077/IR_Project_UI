from flask import Flask, render_template, request
import wikipedia
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial import distance

app = Flask(__name__)

# load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# set device to CUDA if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Returns the BERT embeddings for the given text
def get_bert_embeddings(text):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True, max_length=512)).unsqueeze(0)
    input_ids = input_ids.to(device)
    outputs = model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    embeddings = torch.mean(last_hidden_state, dim=1).squeeze()
    return embeddings.detach().cpu().numpy()

# Returns the top 'n_results' search results from Wikipedia for the given query
def search_wikipedia(query, n_results=5, similarity_threshold=0.5):
    # get search results from Wikipedia API
    search_results = wikipedia.search(query, results=n_results)

    # initialize list to store results
    results = []

    # iterate over search results and get BERT embeddings for each page summary
    try:
        for result in search_results:
            try:
                # get page summary
                page = wikipedia.page(result)
                summary = page.summary

                # get BERT embeddings for query and page summary
                query_embeddings = get_bert_embeddings(query)
                summary_embeddings = get_bert_embeddings(summary)

                # calculate cosine similarity between query and page summary embeddings
                similarity = 1 - (distance.cosine(query_embeddings, summary_embeddings))

                # filter out results with similarity score below threshold
                if similarity < similarity_threshold:
                    continue

                # add result to list
                results.append({'title': page.title, 'url': page.url, 'summary': summary, 'similarity': similarity})

            except wikipedia.exceptions.DisambiguationError as e:
                # if page is disambiguation page, skip it
                continue

    except:
        print("Page Not Found")

    # sort results by similarity in descending order
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    # return top n_results results
    return results[:n_results]

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        results = search_wikipedia(query)
        return render_template('results.html', results=results)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
