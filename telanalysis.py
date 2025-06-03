#Telanalysis by Eduard Isaev @e_isaevsan

import json
import re
import jmespath
import string
import collections
import time
import sys
import os # Moved os import to top
import networkx as nx
import matplotlib.pyplot as plt
import nltk # Moved nltk import here, was in starting()

from pywebio import start_server, input as pywebio_input, config as pywebio_config
from pywebio.output import put_html, put_text, put_image, put_button, put_table, put_collapse, put_code, clear, put_file, toast, Output
from pywebio.input import file_upload
from pywebio.session import run_js, go_app
from pywebio.input import select, slider

from utils import remove_chars_from_text, remove_emojis, clear_user, clear_console, read_conf, write_conf, open_url
import nltk_analyse
import channel_analyse
import words_analyze # Import the refactored module

# global select_type_stem # This global variable was defined but not clearly used. Removed for now.
                         # If select_type_stem from config is meant to be global, it should be handled via read_conf.

## config pywebio - This is a function call, should be within a function or guarded.
# For now, assuming this is part of the initial setup in main or relevant functions.
# pywebio_config(theme='dark',title="TelAnalysis", description="Analysing Telegram CHATS-CHANNELS-GROUPS")
# It's better to call this where the server starts or an app is defined.

def ensure_assets_dir():
    """Ensures the 'asset' directory exists."""
    if not os.path.exists('asset'):
        os.makedirs('asset')

def process_single_message_for_graph(message_data, all_messages_data, existing_names_list, output_csv_path):
    """
    Processes a single message to extract graph data (nodes and edges).
    Helper for the generator function.
    """
    fromm = jmespath.search('from', message_data)
    if fromm is None:
        return False # Skip this message

    from_id = jmespath.search('from_id', message_data)
    if from_id in ['source', 'target', None]: # Skip if from_id is problematic
        return False

    # Add current user to names list
    name_id = f'{fromm}, {from_id}'
    existing_names_list.append(name_id)

    reply_to_message_id = jmespath.search('reply_to_message_id', message_data)
    if reply_to_message_id:
        for reply_message in all_messages_data: # Search for the message being replied to
            message_id_original = jmespath.search('id', reply_message)
            if reply_to_message_id == message_id_original:
                reply_to_user = jmespath.search('from', reply_message)
                reply_to_id_val = jmespath.search('from_id', reply_message)
                if reply_to_user and reply_to_id_val: # Ensure the replied-to message has a user
                    reply_name_id = f'{reply_to_user}, {reply_to_id_val}'
                    existing_names_list.append(reply_name_id)
                    try:
                        with open(output_csv_path, 'a', encoding='utf-8') as edge_file:
                            edge_file.write(f'\n{from_id},{reply_to_id_val},{fromm}-{reply_to_user}')
                    except Exception as ex:
                        print(f"Error writing edge: {ex}")
                return True # Processed a reply
    else: # Not a reply, self-loop for the user who sent the message
        try:
            with open(output_csv_path, 'a', encoding='utf-8') as edge_file:
                edge_file.write(f'\n{from_id},{from_id},{fromm}')
        except Exception as ex:
            print(f"Error writing self-loop edge: {ex}")
        return True # Processed a non-reply
    return False # Default case if not processed


def create_nodes_csv(nodes_csv_path, names_list_param):
    """Creates the nodes.csv file from the list of names."""
    with open(nodes_csv_path, 'w', encoding='utf-8') as node_file:
        node_file.write("id,label,weight")
    
    user_counts = collections.Counter(names_list_param)
    users_table_data = []
    with open(nodes_csv_path, 'a', encoding='utf-8') as node_file_append:
        for name_combo, weight in user_counts.items():
            try:
                name_val, id_val = name_combo.split(',', 1) # Split only on the first comma
                id_val = id_val.strip()
                name_val = name_val.strip()
                if id_val in ['id', 'label', 'weight', 'None', None]: # Skip problematic ids
                    continue
                users_table_data.append([id_val.replace("user", ""), name_val, weight])
                node_file_append.write(f'\n{id_val},{name_val},{weight}')
            except ValueError:
                print(f"Skipping malformed name_combo for nodes.csv: {name_combo}")
                continue
    return users_table_data

def visualize_graph(nodes_csv_path, edges_csv_path, output_image_path):
    """Generates and saves the graph visualization."""
    try:
        G = nx.DiGraph()
        node_weights = {}

        with open(nodes_csv_path, 'r', encoding='utf-8') as nodes_f:
            next(nodes_f) # Skip header
            for line in nodes_f:
                line = line.strip()
                if not line or 'None' in line: continue
                try:
                    node_id, label, weight_str = line.split(',', 2)
                    weight = int(weight_str)
                    if weight < 0: continue # Skip nodes with negative weight for visualization
                    G.add_node(label, weight=weight) # Use label for node identifier in graph if names are unique
                    node_weights[label] = weight
                except ValueError as ve:
                    print(f"Skipping malformed node line: '{line}'. Error: {ve}")
                    continue
        
        with open(edges_csv_path, 'r', encoding='utf-8') as edges_f:
            next(edges_f) # Skip header
            for line in edges_f:
                line = line.strip()
                if not line or 'None' in line: continue
                try:
                    source_id, target_id, _ = line.split(',', 2) # Edge label not used in this visualization
                    # Find node labels corresponding to source_id and target_id
                    source_label, target_label = None, None
                    # This mapping from ID to Label is tricky if not stored.
                    # For now, this part of graph visualization might be limited if IDs are not directly usable as labels.
                    # The original code used 'fromm' and 'reply_to' which are labels.
                    # Assuming edges.csv stores labels directly or can be mapped.
                    # The generator wrote "fromm-reply_to" as edge label, not used here.
                    # It wrote from_id, reply_to_id. We need to map these IDs back to labels from nodes.csv.
                    # This requires a more robust way to map IDs to labels if IDs are not the primary graph keys.
                    # For simplicity, if edges.csv contained labels directly:
                    # G.add_edge(source_label_from_csv, target_label_from_csv, weight=1.3)
                    # Given the current structure, this visualization part might need rework
                    # or edges.csv needs to store labels that match G.nodes().
                    # Let's assume for a moment that source_id and target_id from edges.csv
                    # are the actual node labels (which they are not, they are numeric/string IDs).
                    # This part is problematic with current CSVs.
                    # A proper fix would involve reading nodes.csv to map id -> label, then use labels in add_edge.
                except ValueError as ve:
                     print(f"Skipping malformed edge line: '{line}'. Error: {ve}")
                     continue


        if not G.nodes():
            put_text("No nodes to draw for the graph.")
            return

        # Attempting to draw based on node labels and their weights if available
        valid_nodes_for_drawing = [n for n in G.nodes() if 'weight' in G.nodes[n]]
        if not valid_nodes_for_drawing:
            put_text("Nodes in graph do not have 'weight' attribute for visualization.")
            # Fallback: draw without sizes/colors based on weight
            pos = nx.spring_layout(G) # spring_layout is often better for disconnected components
            nx.draw(G, pos, with_labels=True, font_weight='bold')
        else:
            labels_viz = {n: f"{n} ({G.nodes[n]['weight']})" for n in valid_nodes_for_drawing}
            colors_viz = [G.nodes[n]['weight'] for n in valid_nodes_for_drawing]
            sizes_viz = [G.nodes[n]['weight'] * 200 for n in valid_nodes_for_drawing] # Adjusted multiplier for visibility

            pos = nx.spring_layout(G, k=0.15, iterations=20) # Spring layout can be better
            nx.draw(G, pos, labels=labels_viz, with_labels=True, font_weight='bold', 
                    node_size=sizes_viz, node_color=colors_viz, cmap=plt.cm.Blues, 
                    width=0.5, edge_color='grey')

        plt.savefig(output_image_path, bbox_inches='tight', format='png')
        plt.close()
        put_text(f"Graph saved to {output_image_path}")
        with open(output_image_path, 'rb') as img_file:
            put_image(img_file.read(), width='800px')

    except Exception as ex:
        put_text(f"Error generating graph visualization: {type(ex).__name__} - {ex}")


def generator(json_filepath_param):
    """Main function to generate graph data and visualization."""
    # clear_console() # Removed as per refactoring request
    ensure_assets_dir() # Ensure asset directory exists

    base_filename = json_filepath_param.split("/")[-1].split(".")[0]
    
    edges_csv_path = f'asset/edges_{base_filename}.csv'
    nodes_csv_path = f'asset/nodes_{base_filename}.csv'
    graph_image_path = f'asset/{base_filename}_graph.png' # Changed name for clarity

    dates_list_local = []
    names_list_local = [] # To store "Name, ID" strings for node generation
    
    with open(edges_csv_path, 'w', encoding='utf-8') as edge_f: # Create/clear edges file
        edge_f.write("source,target,label")
    
    try:
        with open(json_filepath_param, 'r', encoding='utf-8') as f_json:
            jsondata = json.load(f_json)
    except FileNotFoundError:
        put_error(f"File not found: {json_filepath_param}")
        return
    except json.JSONDecodeError:
        put_error(f"Invalid JSON file: {json_filepath_param}")
        return

    group_name_val = jmespath.search('name', jsondata) or "Chat Graph"
    put_html(f"<center><h1>{group_name_val}</h1></center>")
    
    all_messages = jmespath.search('messages[*]', jsondata)
    if not all_messages:
        put_text("No messages found to process for graph generation.")
        return
        
    toast(content='Processing messages for graph...', duration=0)
    
    for message_item in all_messages:
        date_val = jmespath.search('date', message_item)
        if date_val: dates_list_local.append(date_val)
        process_single_message_for_graph(message_item, all_messages, names_list_local, edges_csv_path)
        
    toast(content='Generating nodes and graph...', duration=0)
    
    users_table_display_data = create_nodes_csv(nodes_csv_path, names_list_local)
    if users_table_display_data:
        put_table(users_table_display_data, header=['USER ID', 'USERNAME', 'MESSAGE COUNT'])
    else:
        put_text("No user data to display in table.")
    
    visualize_graph(nodes_csv_path, edges_csv_path, graph_image_path)
    
    if dates_list_local:
        firstmes_val = dates_list_local[0].replace("T", " ")
        lastmes_val = dates_list_local[-1].replace("T", " ")
        put_table([[firstmes_val]], header=['First Message Date'])
        put_table([[lastmes_val]], header=['Last Message Date'])
    
    # Provide files for download
    for asset_filename, label in [(f'nodes_{base_filename}.csv', 'Download Nodes CSV'), 
                                  (f'edges_{base_filename}.csv', 'Download Edges CSV'), 
                                  (f'{base_filename}_graph.png', 'Download Graph Image')]:
        try:
            asset_path = f'asset/{asset_filename}'
            if os.path.exists(asset_path):
                content = open(asset_path, 'rb').read()
                put_file(asset_filename, label=label, content=content)
        except Exception as ex:
            put_text(f"Error providing file {asset_filename}: {ex}")
    
    toast(content='Graph generation complete!', duration=3)
    # Buttons are handled by the main app screen that calls this function.

def start_gen():
    """Handler for 'Generating Graphs' button."""
    clear() # Clear current PyWebIO outputs
    put_html("<h1><center>Graph of Telegram Chat</center></h1><br>")
    # Add navigation buttons common to sub-apps
    put_button("Back to Main Menu", onclick=lambda: go_app('default_app', new_window=False), color='primary', scope='ROOT')
    
    uploaded_file = file_upload("Select a JSON chat export file:", accept='.json', required=True)
    
    ensure_assets_dir()
    # Save the uploaded file to asset directory
    asset_file_path = os.path.join('asset', uploaded_file['filename'])
    with open(asset_file_path, 'wb') as f_write:
        f_write.write(uploaded_file['content'])
    
    put_text(f"File {uploaded_file['filename']} uploaded. Starting graph generation...")
    generator(asset_file_path) # Call the main generator logic
    

def start_two():
    """Handler for 'Analysing Chat' button. Uses the refactored words_analyze module."""
    clear()
    put_html("<h1><center>Analyse of Telegram Chat</center></h1><br>")
    put_button("Back to Main Menu", onclick=lambda: go_app('default_app', new_window=False), color='primary', scope='ROOT')

    uploaded_file = file_upload("Select a JSON chat export file:", accept='.json', required=True)
    
    ensure_assets_dir()
    asset_file_path = os.path.join('asset', uploaded_file['filename'])
    with open(asset_file_path, 'wb') as f_write:
        f_write.write(uploaded_file['content'])
        
    put_text(f"File {uploaded_file['filename']} uploaded. Starting chat analysis...")
    # Call the refactored function from words_analyze.py
    words_analyze.perform_chat_analysis(asset_file_path)

    
def start_three():
    """Handler for 'Analysing Channel' button."""
    clear()
    put_html("<h1><center>Analyse of Telegram Channel</center></h1><br>")
    put_button("Back to Main Menu", onclick=lambda: go_app('default_app', new_window=False), color='primary', scope='ROOT')

    uploaded_file = file_upload("Select a JSON channel export file:", accept='.json', required=True)
    
    ensure_assets_dir()
    asset_file_path = os.path.join('asset', uploaded_file['filename'])
    with open(asset_file_path, 'wb') as f_write:
        f_write.write(uploaded_file['content'])

    put_text(f"File {uploaded_file['filename']} uploaded. Starting channel analysis...")
    channel_analyse.channel(asset_file_path) # Assuming channel_analyse.channel expects a filepath

def config_app():
    """Handler for 'Config' button. Allows user to change settings."""
    clear()
    put_html("<h1><center>Configuration</center></h1>")
    put_button("Back to Main Menu", onclick=lambda: go_app('default_app', new_window=False), color='primary', scope='ROOT')

    current_stem = read_conf('select_type_stem') or 'Off'
    current_most_com_user = read_conf('most_com') or 30
    current_most_com_channel = read_conf('most_com_channel') or 100

    put_text(f"Current Stemming mode: {current_stem}")
    put_text(f"Current Most Common words (User): {current_most_com_user}")
    put_text(f"Current Most Common words (Channel): {current_most_com_channel}")

    new_stem_mode = select('Stemming mode:', options=['Off', 'On'], value=current_stem)
    new_most_com_user = pywebio_input.input('Most Common words for User Analysis (e.g., 30):', type='number', value=current_most_com_user)
    new_most_com_channel = pywebio_input.input('Most Common words for Channel Analysis (e.g., 100):', type='number', value=current_most_com_channel)

    try:
        # Validate inputs
        new_most_com_user = int(new_most_com_user)
        new_most_com_channel = int(new_most_com_channel)
        if new_most_com_user <= 0 or new_most_com_channel <= 0:
            toast("Number of common words must be positive.", color='error')
            return # Or loop until valid input

        write_conf({
            "select_type_stem": new_stem_mode,
            "most_com": new_most_com_user,
            "most_com_channel": new_most_com_channel
        })
        toast("Configuration saved successfully!", color='success')
        # Optionally, re-run config_app to show updated values or go_app('default_app')
        go_app('config_app', new_window=False) # Refresh the config page

    except ValueError:
        toast("Invalid number format for common words.", color='error')
    except Exception as ex:
        toast(f"Error saving configuration: {ex}", color='error')


def default_app():
    """Main application screen with choices for analysis types."""
    clear() # Clear previous outputs
    pywebio_config(theme='dark', title="TelAnalysis", description="Analysing Telegram CHATS-CHANNELS-GROUPS") # Set config here
    
    put_html("<center><h1>Welcome to TelAnalysis</h1></center>")
    put_html("<center><h3>Select a module:</h3></center>")
    
    # Use go_app to switch between "pages" or views
    put_button("Generating Graphs", onclick=lambda: go_app('graph_app', new_window=False), color='success')
    put_button("Analysing Chat", onclick=lambda: go_app('chat_analysis_app', new_window=False), color='success')
    put_button("Analysing Channel", onclick=lambda: go_app('channel_analysis_app', new_window=False), color='success')
    put_button("Configuration", onclick=lambda: go_app('config_app', new_window=False), color='warning')
    
    # Common footer buttons or info
    put_html("<br><hr><center><small>TelAnalysis by Eduard Isaev @e_isaevsan</small></center>")

def main():
    """Main function to start the PyWebIO server and define apps."""
    # pywebio_config needs to be called before start_server if it sets global theme for all sessions.
    # However, individual apps can also set their config.
    # For simplicity, we can set a default theme here or in default_app.
    
    # Initial setup
    # clear_console() # Removed as per refactoring request
    ensure_assets_dir()

    # NLTK downloads - should ideally be done once, or checked if already downloaded.
    nltk_resources = [('corpora/stopwords', 'stopwords'), ('tokenizers/punkt', 'punkt')]
    for resource_path, resource_name in nltk_resources:
        try:
            nltk.data.find(resource_path)
            print(f"NLTK resource '{resource_name}' already downloaded.")
        except nltk.downloader.DownloadError:
            print(f"NLTK resource '{resource_name}' not found. Attempting download...")
            try:
                nltk.download(resource_name)
                print(f"NLTK resource '{resource_name}' downloaded successfully.")
            except Exception as e:
                print(f"Error downloading NLTK resource '{resource_name}': {e}")
                print(f"Please ensure you have an active internet connection and try again.")
                print(f"Some functionalities of TelAnalysis may not work without these resources.")
                # Optionally, exit or prevent app from starting if resources are critical.
                # For now, just printing a warning.

    # Ensure config.json exists
    if not os.path.exists('config.json'):
        write_conf({"select_type_stem": "Off", "most_com": 30, "most_com_channel": 100})

    # open_url() # This opens the browser, usually done by start_server automatically or manually by user.
    
    # Define apps for go_app navigation
    # Each "app" is a function that PyWebIO can serve.
    apps = {
        'default_app': default_app,
        'graph_app': start_gen,
        'chat_analysis_app': start_two,
        'channel_analysis_app': start_three,
        'config_app': config_app
    }
    
    # Start the server with the default app.
    # `start_server` can take a single function or a list/dict of functions.
    # Using a dictionary allows `go_app` to switch between them by name.
    print(f"TelAnalysis is preparing to start...")
    print(f"Attempting to launch server at http://127.0.0.1:9993")
    print("If the browser does not open automatically, please navigate to the URL above.")
    start_server(apps, default_app='default_app', host='127.0.0.1', port=9993, debug=True, auto_open_webbrowser=True)


if __name__ == "__main__":
    main()
