import numpy as np
import pandas as pd

df = pd.read_csv('My Spotify Library.csv')
titles = df.iloc[1:, 0].to_numpy() #List to store music titles
artists = df.iloc[1:, 1].to_numpy() #List to store music authors
spotify_ids = df.iloc[1:, 6].to_numpy() #List to store Spotify IDs
music_dict = {title: i for i, title in enumerate(titles)} #Dictionary to
n = len(titles) #Total number of musics

criteria = np.array(["Pop", "Rap", "Blues - Jazz", "Country - Rock - Métal",
                     "Disco, House, Electro, Techno", "Chill - Energique Agressif"])
d = len(criteria) #Total number of criteria


def slct_music_idx(m):
    return np.random.randint(0, n, size=m)
def music_titles(indexes):
    return [titles[i] for i in indexes]
def music_artists(indexes):
    return [artists[i] for i in indexes]


def test():
    print(f"n = {n}")
    print(f"Titles: {titles}")
    print(f"Artists: {artists}")
    print(f"d = {d}")
# test()