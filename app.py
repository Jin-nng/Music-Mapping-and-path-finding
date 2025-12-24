from flask import Flask, render_template, request
import data
import csv
import path_finding
import os

app = Flask(__name__)
def home_page():
    indexes = data.slct_music_idx(10)
    music_titles = data.music_titles(indexes)
    music_authors = data.music_artists(indexes)
    return render_template('home.html', music_titles=music_titles, music_authors=music_authors, zip=zip)

# Route for home page
@app.route('/')
def home():
    return home_page()

# Route for dynamic music pages
@app.route('/<string:music_name>')
def music_page(music_name: str):
    music_spotify_id = data.spotify_ids[data.music_dict[music_name]]
    return render_template('music_page.html', 
                           music_name_html=music_name, 
                           music_author_html=data.artists[data.music_dict[music_name]],
                           music_spotify_id=music_spotify_id)

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    music_name = request.form['music_name']
    pop = request.form['pop']
    rap = request.form['rap']
    jazz = request.form['jazz']
    rock = request.form['rock'] 
    electro = request.form['electro']
    chill = request.form['chill']

    # Update the occurences list
    path_finding.occurences[data.music_dict[music_name]] += 1
        
    # Save to CSV
    file_exists = os.path.isfile('responses.csv')
    with open('responses.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['music_name', 'pop', 'rap', 'jazz', 'rock', 'electro', 'chill'])
        writer.writerow([music_name, pop, rap, jazz, rock, electro, chill])

    return home_page()