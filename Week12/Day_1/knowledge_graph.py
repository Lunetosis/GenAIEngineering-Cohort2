import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data into a DataFrame
movie_data = pd.read_csv('imdb.csv').head(200)

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes and edges
for index, row in movie_data.iterrows():
    movie_title = row['Title']
    director = row['Director']
    genres = row['Genre'].split(',')
    actors = row['Actors'].split(',')
    rating = row['Rating']
    revenue = pd.to_numeric(row['Revenue (Millions)'], errors='coerce')
    
    # Add movie node
    G.add_node(movie_title, type='movie', rating=rating, revenue=revenue)

    # Connect directors to movies
    G.add_node(director, type='director')
    G.add_edge(director, movie_title, weight=rating)

    # Connect genres to movies
    for genre in genres:
        G.add_node(genre, type='genre')
        G.add_edge(genre, movie_title, weight=rating)

    # Connect actors to movies
    for actor in actors:
        G.add_node(actor.strip(), type='actor')
        G.add_edge(actor.strip(), movie_title, weight=rating)

# Define color map based on node types
color_map = []
for node in G:
    if G.nodes[node]['type'] == 'movie':
        color_map.append('green')
    elif G.nodes[node]['type'] == 'director':
        color_map.append('blue')
    elif G.nodes[node]['type'] == 'genre':
        color_map.append('orange')
    else:
        color_map.append('red')

# Drawing the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, node_color=color_map, with_labels=True, font_size=8, node_size=500)
plt.title('IMDB Movie Knowledge Graph')
plt.savefig('imdb_knowledge_graph.png')
plt.show()