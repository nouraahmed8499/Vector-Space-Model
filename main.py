import VSM
VSM.initialize_terms_and_postings()
VSM.initialize_document_frequencies()
VSM.norm()
from flask import Flask, render_template, request, redirect, url_for

app= Flask(__name__)
@app.route('/')
def dictionary():
    return render_template('index.html')

#Funtion will invoke whenever a query is posted
@app.route("/query", methods=['POST'])
def query():
    #getting query from the HTML form
    query = request.form['query']
    if query:
        result = VSM.do_search(query)
        documents_content = VSM.documents_content
        if result !=0:
            return render_template('dictionary.html',res = result,content=documents_content, num= len(result))
        else: return render_template('tryAgain.html')
    else : return render_template('index.html')
if __name__ == '__main__':
    app.run()
