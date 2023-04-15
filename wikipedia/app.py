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
def search_wikipedia(query, n_results=5, similarity_threshold=0.5, feedback=None):
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

    if feedback:
        for i, result in enumerate(results):
            if i not in feedback:
                continue
            if feedback[i] == 0:
                results[i]['similarity'] = 0

    # return top n_results results
    return results[:n_results]

def get_user_feedback(results):
    feedback = {}
    for i, result in enumerate(results):
        title = result['title']
        feedback[title] = False
    return feedback

def refine_search_query(query, results, feedback):
    new_query = query
    for i, result in enumerate(results):
        if i not in feedback:
            continue
        if feedback[i] == 0:
            continue
        elif feedback[i] == 1:
            new_query = result['title']
            break
    return new_query

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        results = search_wikipedia(query)

        results = search_wikipedia(query)

        # ask the user for feedback and refine the search query
        feedback = get_user_feedback(results)
        new_query = refine_search_query(query, results, feedback)

        # do the refined search
        results = search_wikipedia(new_query, feedback=feedback)

        while len(results) == 0:
            print("No results found. Please refine your search query.")
            feedback = get_user_feedback([])
            new_query = refine_search_query(query, results, feedback=feedback)
            results = search_wikipedia(new_query)

        return render_template('results.html', results=results, feedback=feedback)

    else:
        return render_template('index.html')

import json
from urllib.parse import quote
@app.route('/refine-search', methods=['POST'])
def refine_search():
    query = request.form['query']
    results = search_wikipedia(query)

    # ask the user for feedback and refine the search query
    feedback = get_user_feedback(results)
    new_query = refine_search_query(query, results, feedback)

    # do the refined search
    results = search_wikipedia(new_query, feedback=feedback)

    while len(results) == 0:
        print("No results found. Please refine your search query.")
        feedback = get_user_feedback([])
        new_query = refine_search_query(query, results, feedback=feedback)
        results = search_wikipedia(new_query)

    # convert results to JSON string and pass as parameter in URL
    results_json = json.dumps(results)
    results_encoded = quote(results_json)
    return redirect(url_for('refined_results', query=new_query, results=results_encoded))

from urllib.parse import unquote
@app.route('/refined_results.html', methods=['POST'])
def refined_results():
    selected_results = []
    for key in request.form:
        if key.startswith('result'):
            selected_results.append(request.form[key])
    results = search_wikipedia(selected_results)
    return render_template('refined_results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
