from utils import remove_chars_from_text, remove_emojis, clear_user, read_conf
import nltk_analyse
import sys
from pywebio import config, output, pin, session
import json, re, jmespath
from validate_email import validate_email
import phonenumbers
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Action map for different message types
ACTION_MAP = {
    'invite_members': 'Invite Member',
    'remove_members': 'Kicked Members',
    'join_group_by_link': 'Joined by Link',
    'pin_message': 'Pinned Message',
    # Add other actions as needed
}

# Initializing the sentiment analyzer
ANALYZER = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    try:
        score = ANALYZER.polarity_scores(str(text))
        return float(score['compound'])  # Converting to float for certainty
    except:
        return float(0.0)

def extract_emails_and_phone_numbers(text):
    """Extracts email addresses and phone numbers from a given text."""
    emails_list = []
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    for email in emails:
        if validate_email(email, verify=False):
            emails_list.append(email)
    phones_list = []
    phone_numbers = re.findall(r'\+?[0-9]{1,3}?[-. (]?[0-9]{1,4}[-. )]?[0-9]{1,4}[-. ]?[0-9]{1,9}', text)
    for phones in phone_numbers:
        try:
            phone_number = phonenumbers.parse(phones, None)
            if phonenumbers.is_valid_number(phone_number):
                phones_list.append(phones)
        except Exception:
            pass
    return emails_list, phones_list

def extract_text_from_message(message_obj):
    """
    Recursively extracts all unique text strings from a Telegram message object.

    Telegram messages can have text in various fields:
    - 'text': Plain text of the message.
    - 'caption': Caption for media (photos, videos, documents).
    - 'text_entities': Formatted text parts (bold, italic, links with text, etc.).
                       The 'text' field within each entity is extracted.
    - Nested messages: 'forwarded_from' (original message) and 'reply_to_message'
                       are recursively processed.
    - Generic search: The function also iterates through all dictionary values and
                      list items in the message object, recursively calling itself
                      if it finds nested lists or dictionaries. This helps capture
                      text from less common or custom message structures.

    Args:
        message_obj: A dictionary or list representing a part of or the entire
                     Telegram message structure.

    Returns:
        A set of unique text strings found within the message object.
        Using a set ensures that duplicate text segments are not returned.
    """
    texts_set = set()  # Using a set for unique text values

    if isinstance(message_obj, dict):
        if 'text' in message_obj:
            if isinstance(message_obj['text'], str) and message_obj['text'].strip():
                texts_set.add(message_obj['text'])
            elif isinstance(message_obj['text'], list):
                for item in message_obj['text']:
                    if isinstance(item, str): # Ensure item is a string
                        texts_set.add(item)
        
        if 'caption' in message_obj:
            if isinstance(message_obj['caption'], str) and message_obj['caption'].strip():
                texts_set.add(message_obj['caption'])
        
        entities = jmespath.search('text_entities[*].text', message_obj)
        if entities:
            for entity in entities:
                if isinstance(entity, str): # Ensure entity is a string
                    texts_set.add(entity)

        if 'forwarded_from' in message_obj: # Check if key exists
            texts_set.update(extract_text_from_message(message_obj['forwarded_from']))

        if 'reply_to_message' in message_obj: # Check if key exists
            texts_set.update(extract_text_from_message(message_obj['reply_to_message']))

        for key, value in message_obj.items():
            # Avoid re-processing already specifically handled keys if they are also dicts/lists.
            # However, the current broad recursion is simple and `set` handles duplicates.
            # For very deep or large objects, optimizing this might be needed, but likely fine for messages.
            if key not in ['text', 'caption', 'text_entities', 'forwarded_from', 'reply_to_message']: # Avoid redundant processing of top-level known fields
                if isinstance(value, (list, dict)): 
                    texts_set.update(extract_text_from_message(value))

    elif isinstance(message_obj, list):
        for item_in_list in message_obj:
            texts_set.update(extract_text_from_message(item_in_list))

    return texts_set

def process_message_data(message_item, current_users_data, current_emails_list, current_phones_list):
    """Processes a single message and updates user data, emails, and phone numbers."""
    user_id_val = jmespath.search('from_id', message_item)
    action_text_val = ""
    message_count_increment = 0

    if not user_id_val:
        user_id_val = jmespath.search('actor_id', message_item)
        if user_id_val:
            user_id_val = user_id_val.replace(" ", "")
            if user_id_val not in current_users_data:
                current_users_data[user_id_val] = []

            action = jmespath.search('action', message_item)
            if action:
                tex = jmespath.search('text', message_item) or ''
                action_text_val = ACTION_MAP.get(action, action)
                
                if action in ['invite_members', 'remove_members']:
                    members = jmespath.search('members', message_item)
                    members_str = ",".join(str(x) for x in members if x)
                    current_users_data[user_id_val].append((f"{action_text_val} - {members_str}", 0.0))
                else:
                    current_users_data[user_id_val].append((f"{action_text_val} {tex}", 0.0))
                return current_users_data, current_emails_list, current_phones_list, message_count_increment

    if not user_id_val: 
        return current_users_data, current_emails_list, current_phones_list, message_count_increment

    user_id_val = user_id_val.replace(" ", "")
    if user_id_val not in current_users_data:
        current_users_data[user_id_val] = []
    
    message_count_increment = 1

    unique_texts = extract_text_from_message(message_item)
    for clean_text in unique_texts:
        if clean_text: # Ensure clean_text is not empty
            sentiment_score = analyze_sentiment(clean_text)
            current_users_data[user_id_val].append((clean_text, sentiment_score))
            
            extracted_emails, extracted_phone_numbers = extract_emails_and_phone_numbers(clean_text)
            current_emails_list.extend(extracted_emails)
            current_phones_list.extend(extracted_phone_numbers)
            
    return current_users_data, current_emails_list, current_phones_list, message_count_increment


def perform_chat_analysis(json_file_path_param):
    """
    Performs chat analysis on a given JSON file.
    Handles PyWebIO outputs for displaying results.
    """
    # Encapsulated global variables
    local_emails, local_phoness, local_all_tokens, local_users = [], [], [], {}
    local_count_messages = 0

    # Interface configuration
    config(theme='dark', title="TelAnalysis", description="Analysing Telegram CHATS-CHANNELS-GROUPS")
    output.toast(content='Wait..', duration=2)
    output.put_button("Scroll Down", onclick=lambda: session.run_js('window.scrollTo(0, document.body.scrollHeight)'))
    output.put_button("Close", onclick=lambda: session.run_js('window.close()'), color='danger')
    output.put_html("<h1><center>Analyse of Telegram Chat<center></h1><br>")

    # User ID input
    pin.put_input('ID')
    output.put_button("Search ID", onclick=lambda: session.run_js(f'window.find({pin.pin.ID}, true)'), color='warning')
    
    # Extracting filename for asset creation if needed, assuming json_file_path is like 'asset/filename.json'
    filename_for_asset = json_file_path.split("/")[-1].split(".")[0]


    # Ensuring the file is opened with the correct encoding
    with open(json_file_path, 'r', encoding='utf-8', errors='replace') as datas:
        data_content = json.load(datas)

    sf_messages = jmespath.search('messages[*]', data_content)
    group_name = jmespath.search('name', data_content)

    # Processing messages using thread pooling
    with ThreadPoolExecutor() as executor:
        futures = []
        for msg in sf_messages:
            futures.append(executor.submit(process_message_data, msg, local_users, local_emails, local_phoness))

        for future in as_completed(futures):
            try:
                # Update local_users, local_emails, local_phoness with results
                # and increment local_count_messages
                processed_users, processed_emails, processed_phoness, msg_increment = future.result()
                local_users.update(processed_users) # Note: this simple update might not be ideal if keys overlap; consider a merge strategy
                local_emails.extend(processed_emails) # Extend works as lists are mutable
                local_phoness.extend(processed_phoness)
                local_count_messages += msg_increment
            except Exception as e:
                print(f"Error processing message future: {e}")


    # Now displaying in the desired format user - user_from
    for user_id, da_messages in local_users.items():
        user_from = ""
        # Search for the 'from' field using the user_id
        for m in sf_messages: # Iterate through original messages to find 'from'
            if jmespath.search('from_id', m) == user_id:
                user_from = jmespath.search('from', m)
                break
            elif jmespath.search('actor_id', m) == user_id: # Fallback for actor_id
                user_from = jmespath.search('actor', m) # Assuming 'actor' field exists for actor_id
                if not user_from: user_from = user_id # Default to ID if no name
                break
        
        user_display_name = f"{user_from} - {user_id}" if user_from else user_id
            
        user_sentiment_scores = [float(x[1]) for x in da_messages if isinstance(x[1], (float, int))] # Allow int scores too
        average_user_sentiment = sum(user_sentiment_scores) / len(user_sentiment_scores) if user_sentiment_scores else 0.0
        
        try:
            most_com_val = read_conf('most_com')
            # Pass only message texts for NLTK analysis
            message_texts_for_nltk = [str(x[0]) for x in da_messages]
            genuy, tokens = nltk_analyse.analyse(message_texts_for_nltk, most_com_val)

            gemy = [[x, y] for x, y in genuy]
            # gery was [[x[0]] for x in da_messages], it seems to be just the messages, already in da_messages
            local_all_tokens.extend(tokens)

            if da_messages or gemy: # Check if there are messages or words to display
                output.put_collapse(user_display_name, [
                    f'Messages of {user_display_name}',
                    output.put_text(f'Average Sentiment for {user_display_name}: {average_user_sentiment:.2f}'),
                    output.put_table([[x[0], f"{x[1]:.2f}"] for x in da_messages], header=['Messages', 'Sentiment Score']), # Format score
                    output.put_table(gemy, header=['Word', 'Count'])
                ], open=False)

        except Exception as ex:
            print(f"Error during user output for {user_display_name}: {ex}")
            output.put_text(f"Could not display analysis for {user_display_name} due to an error.")


    # Overall analysis of all messages
    most_com_overall = read_conf('most_com')
    
    # Ensure local_all_tokens contains strings for analyse_all
    # NLTK analyse_all expects a list of tokens (words)
    # local_all_tokens should already be a list of words from previous steps.

    if local_all_tokens: # Check if there are any tokens to analyze
        try:
            # analyse_all returns: fdist (list of [word, count]), data (list of words)
            fdist_all, data_words_all = nltk_analyse.analyse_all(local_all_tokens, most_com_overall)
            
            if fdist_all: # If fdist_all has content
                all_chat_table_data = [[word, count] for word, count in fdist_all]
                output.put_collapse(f'TOP words of {group_name}', [
                    output.put_table(all_chat_table_data, header=['Word', 'Count']),
                ], open=False)

                # Overall chat sentiment analysis using the raw words from `data_words_all`
                # This assumes analyse_sentiment can process a list of words; it's designed for sentences/phrases.
                # A more accurate overall sentiment might involve averaging user sentiments or message sentiments.
                # For now, let's try to get sentiment from the most common words joined, or average message sentiments.
                # Re-evaluate: It's better to average sentiments of all messages if possible.
                # However, `all_tokens` in the original code was a list of words.
                # Let's average the sentiment of the top words as a proxy if direct message texts aren't available here.
                # This part might need further refinement for accuracy.
                
                # Option 1: Sentiment of joined top words (less accurate)
                # text_for_sentiment_all = " ".join(data_words_all)
                # average_chat_sentiment = analyze_sentiment(text_for_sentiment_all)
                
                # Option 2: Average of all user average sentiments (more representative of overall chat mood)
                all_user_average_sentiments = []
                for _, da_msgs in local_users.items():
                    user_sent_scores = [float(x[1]) for x in da_msgs if isinstance(x[1], (float, int))]
                    if user_sent_scores:
                        all_user_average_sentiments.append(sum(user_sent_scores) / len(user_sent_scores))
                
                if all_user_average_sentiments:
                    average_chat_sentiment = sum(all_user_average_sentiments) / len(all_user_average_sentiments)
                    output.put_text(f'Average Chat Sentiment for {group_name}: {average_chat_sentiment:.2f}')
                else:
                    output.put_text(f"Overall sentiment for {group_name} could not be calculated (no user sentiments).")

            else:
                output.put_text(f"No common words found for overall analysis in {group_name}.")

        except Exception as e:
            print(f"Error in overall NLTK analysis: {e}")
            output.put_text(f"Could not perform overall word analysis for {group_name}.")
    else:
        output.put_text(f"No tokens found for overall analysis in {group_name}. Sentiment analysis is unavailable.")


    # Processing emails and phones
    unique_emails = [[email] for email in set(local_emails)]
    unique_phoness = [[ph] for ph in set(local_phoness)]
    output.put_collapse('Found Emails and Numbers', [
        output.put_table(unique_emails, header=['Emails:']),
        output.put_table(unique_phoness, header=['Numbers:'])
    ], open=False)

    # Additional buttons
    output.put_button("Close", onclick=lambda: session.run_js('window.close()'), color='danger')
    output.put_button("Scroll Up", onclick=lambda: session.run_js('window.scrollTo(document.body.scrollHeight, 0)'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        # This part is tricky because PyWebIO apps are usually started with a function target,
        # not by running a script that directly calls PyWebIO output functions.
        # For direct script execution to work with PyWebIO, you'd typically wrap perform_chat_analysis
        # in a main PyWebIO function if it's the entry point of a PyWebIO app.
        # However, the goal here is to make perform_chat_analysis callable from telanalysis.py.
        # So, this __main__ block is more for testing or direct invocation if needed,
        # but the primary use will be importing perform_chat_analysis.
        
        # To make this runnable as a script for testing with PyWebIO, we'd need a dummy server start:
        from pywebio import start_server
        def main_pywebio_app():
            perform_chat_analysis(json_file_path)
        
        # This would start a server just for this analysis.
        # This is likely NOT what's intended for integration with telanalysis.py,
        # which already manages its own PyWebIO server and app flow.
        # For now, let's assume direct calls to perform_chat_analysis will be made
        # within an existing PyWebIO session managed by telanalysis.py.
        # The PyWebIO config and output calls within perform_chat_analysis will then
        # target the currently active session.
        
        # If running this script directly:
        # print(f"Analyzing {json_file_path}...")
        # perform_chat_analysis(json_file_path) 
        # This won't work as expected without a PyWebIO server context.
        # The refactoring assumes perform_chat_analysis is called within
        # a PyWebIO app managed by telanalysis.py.
        pass
    else:
        print("Usage: python words_analyze.py <path_to_json_file>")

```python
from utils import remove_chars_from_text, remove_emojis, clear_user, read_conf
import nltk_analyse
import sys
from pywebio import config, output, pin, session
import json, re, jmespath
from validate_email import validate_email
import phonenumbers
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Action map for different message types
ACTION_MAP = {
    'invite_members': 'Invite Member',
    'remove_members': 'Kicked Members',
    'join_group_by_link': 'Joined by Link',
    'pin_message': 'Pinned Message',
    # Add other actions as needed
}

# Initializing the sentiment analyzer
ANALYZER = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    try:
        score = ANALYZER.polarity_scores(str(text))
        return float(score['compound'])  # Converting to float for certainty
    except:
        return float(0.0)

def extract_emails_and_phone_numbers(text):
    """Extracts email addresses and phone numbers from a given text."""
    emails_list = []
    emails_re = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    for email_val in emails_re:
        if validate_email(email_val, verify=False):
            emails_list.append(email_val)
    phones_list = []
    phone_numbers_re = re.findall(r'\+?[0-9]{1,3}?[-. (]?[0-9]{1,4}[-. )]?[0-9]{1,4}[-. ]?[0-9]{1,9}', text)
    for phones_val in phone_numbers_re:
        try:
            phone_number = phonenumbers.parse(phones_val, None)
            if phonenumbers.is_valid_number(phone_number):
                phones_list.append(phones_val)
        except Exception:
            pass
    return emails_list, phones_list

def extract_text_from_message(message_obj):
    """Extracts text from a message object, handling nested structures."""
    texts_set = set()  # Using a set for unique text values

    if isinstance(message_obj, dict):
        if 'text' in message_obj:
            if isinstance(message_obj['text'], str) and message_obj['text'].strip():
                texts_set.add(message_obj['text'])
            elif isinstance(message_obj['text'], list):
                for item in message_obj['text']:
                    if isinstance(item, str):
                        texts_set.add(item)
        
        if 'caption' in message_obj:
            if isinstance(message_obj['caption'], str) and message_obj['caption'].strip():
                texts_set.add(message_obj['caption'])
        
        entities = jmespath.search('text_entities[*].text', message_obj)
        if entities:
            for entity in entities:
                texts_set.add(entity)

        if 'forwarded_from' in message_obj:
            texts_set.update(extract_text_from_message(message_obj['forwarded_from']))

        if 'reply_to_message' in message_obj:
            texts_set.update(extract_text_from_message(message_obj['reply_to_message']))

        for key, value in message_obj.items():
            if isinstance(value, (list, dict)): # Check if value is list or dict before recursive call
                texts_set.update(extract_text_from_message(value))

    elif isinstance(message_obj, list):
        for item_in_list in message_obj:
            texts_set.update(extract_text_from_message(item_in_list))

    return texts_set

def process_message_data(message_item, current_users_data, current_emails_list, current_phones_list):
    """Processes a single message and updates user data, emails, and phone numbers."""
    user_id_val = jmespath.search('from_id', message_item)
    action_text_val = ""
    message_count_increment = 0

    if not user_id_val:
        user_id_val = jmespath.search('actor_id', message_item)
        if user_id_val:
            user_id_val = user_id_val.replace(" ", "")
            if user_id_val not in current_users_data:
                current_users_data[user_id_val] = []

            action = jmespath.search('action', message_item)
            if action:
                tex = jmespath.search('text', message_item) or ''
                action_text_val = ACTION_MAP.get(action, action)
                
                if action in ['invite_members', 'remove_members']:
                    members = jmespath.search('members', message_item)
                    members_str = ",".join(str(x) for x in members if x)
                    current_users_data[user_id_val].append((f"{action_text_val} - {members_str}", 0.0))
                else:
                    current_users_data[user_id_val].append((f"{action_text_val} {tex}", 0.0))
                return current_users_data, current_emails_list, current_phones_list, message_count_increment

    if not user_id_val: 
        return current_users_data, current_emails_list, current_phones_list, message_count_increment

    user_id_val = user_id_val.replace(" ", "")
    if user_id_val not in current_users_data:
        current_users_data[user_id_val] = []
    
    message_count_increment = 1

    unique_texts = extract_text_from_message(message_item)
    for clean_text in unique_texts:
        if clean_text: # Ensure clean_text is not empty
            sentiment_score = analyze_sentiment(clean_text)
            current_users_data[user_id_val].append((clean_text, sentiment_score))
            
            extracted_emails, extracted_phone_numbers = extract_emails_and_phone_numbers(clean_text)
            current_emails_list.extend(extracted_emails)
            current_phones_list.extend(extracted_phone_numbers)
            
    return current_users_data, current_emails_list, current_phones_list, message_count_increment


def perform_chat_analysis(json_file_path_param):
    """
    Performs chat analysis on a given JSON file.
    Handles PyWebIO outputs for displaying results.
    """
    local_emails_list, local_phones_list, local_all_tokens_list, local_users_data = [], [], [], {}
    local_message_count = 0

    # Interface configuration (can be set by the calling app, e.g., telanalysis.py)
    # config(theme='dark', title="TelAnalysis", description="Analysing Telegram CHATS-CHANNELS-GROUPS") 
    # output.toast(content='Wait..', duration=2) # This might be better handled by the caller
    
    # These buttons are part of the main app flow, less so for a callable function.
    # output.put_button("Scroll Down", onclick=lambda: session.run_js('window.scrollTo(0, document.body.scrollHeight)'))
    # output.put_button("Close", onclick=lambda: session.run_js('window.close()'), color='danger')
    output.put_html("<h1><center>Analyse of Telegram Chat<center></h1><br>")

    pin.put_input('ID_search_words_analyze', label='Search by User ID in results:') # Changed pin ID to be specific
    output.put_button("Search ID in Results", onclick=lambda: session.run_js(f'window.find(pin.pin.ID_search_words_analyze, true)'), color='warning')
    
    with open(json_file_path_param, 'r', encoding='utf-8', errors='replace') as data_file:
        data_content = json.load(data_file)

    sf_messages = jmespath.search('messages[*]', data_content)
    group_name = jmespath.search('name', data_content) or "Chat" # Default group name

    # Use a temporary dictionary for users_data within the ThreadPoolExecutor
    # to avoid issues with concurrent modifications if local_users_data was directly passed and modified.
    # This is a simplified approach; for complex scenarios, thread-safe data structures or more careful handling would be needed.
    # However, process_message_data is designed to return new states, which helps.

    temp_users_data_map = {msg_idx: {} for msg_idx, _ in enumerate(sf_messages)} # To store results per message if needed, or just aggregate
    aggregated_emails = []
    aggregated_phoness = []
    
    with ThreadPoolExecutor(max_workers=10) as executor: # Limit workers if needed
        future_to_idx = {}
        for idx, msg_item in enumerate(sf_messages):
            # Each task gets its own copy of lists to extend, to avoid race conditions on global_emails/phoness
            future = executor.submit(process_message_data, msg_item, {}, [], []) 
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                processed_users, processed_emails, processed_phoness, msg_increment = future.result()
                
                # Merge results:
                for user_id, messages in processed_users.items():
                    if user_id not in local_users_data:
                        local_users_data[user_id] = []
                    local_users_data[user_id].extend(messages)
                
                aggregated_emails.extend(processed_emails)
                aggregated_phoness.extend(processed_phoness)
                local_message_count += msg_increment
            except Exception as e:
                output.put_text(f"Error processing message index {idx}: {e}")
    
    local_emails_list.extend(list(set(aggregated_emails))) # Make unique
    local_phones_list.extend(list(set(aggregated_phoness))) # Make unique


    for user_id_key, user_messages_list in local_users_data.items():
        user_from_name = ""
        for m_item in sf_messages: 
            if jmespath.search('from_id', m_item) == user_id_key:
                user_from_name = jmespath.search('from', m_item)
                break
            elif jmespath.search('actor_id', m_item) == user_id_key: 
                user_from_name = jmespath.search('actor', m_item) 
                if not user_from_name: user_from_name = user_id_key 
                break
        
        user_display_name = f"{user_from_name} ({user_id_key})" if user_from_name else user_id_key
            
        user_sentiment_scores = [float(x[1]) for x in user_messages_list if isinstance(x[1], (float, int))]
        average_user_sentiment = sum(user_sentiment_scores) / len(user_sentiment_scores) if user_sentiment_scores else 0.0
        
        try:
            most_com_val = read_conf('most_com')
            message_texts_for_nltk = [str(x[0]) for x in user_messages_list if x[0]] # Ensure text is not None
            
            if message_texts_for_nltk: # Only run analyse if there's text
                genuy_freq_dist, tokens_list = nltk_analyse.analyse(message_texts_for_nltk, most_com_val)
                gemy_table_data = [[word, count] for word, count in genuy_freq_dist]
                local_all_tokens_list.extend(tokens_list)
            else:
                gemy_table_data = []


            if user_messages_list or gemy_table_data: 
                output.put_collapse(user_display_name, [
                    output.put_text(f'Messages of {user_display_name} (Count: {len(user_messages_list)})'),
                    output.put_text(f'Average Sentiment for {user_display_name}: {average_user_sentiment:.2f}'),
                    output.put_table([[x[0], f"{x[1]:.2f}"] for x in user_messages_list], header=['Message', 'Sentiment Score']),
                    output.put_table(gemy_table_data, header=['Word', 'Count'])
                ], open=False)

        except Exception as ex:
            output.put_text(f"Error during user output for {user_display_name}: {type(ex).__name__} - {ex}")

    most_com_overall = read_conf('most_com')
    if local_all_tokens_list:
        try:
            fdist_all, data_words_all = nltk_analyse.analyse_all(local_all_tokens_list, most_com_overall)
            
            if fdist_all:
                all_chat_table_data = [[word, count] for word, count in fdist_all]
                output.put_collapse(f'TOP words of {group_name} (Total analyzed tokens: {len(local_all_tokens_list)})', [
                    output.put_table(all_chat_table_data, header=['Word', 'Count']),
                ], open=True) # Open by default for overall summary

                all_user_average_sentiments = []
                for _, user_msgs in local_users_data.items():
                    user_sent_scores = [float(x[1]) for x in user_msgs if isinstance(x[1], (float, int))]
                    if user_sent_scores:
                        all_user_average_sentiments.append(sum(user_sent_scores) / len(user_sent_scores))
                
                if all_user_average_sentiments:
                    average_chat_sentiment = sum(all_user_average_sentiments) / len(all_user_average_sentiments)
                    output.put_text(f'Average Chat Sentiment for {group_name}: {average_chat_sentiment:.2f}')
                else:
                    output.put_text(f"Overall sentiment for {group_name} could not be calculated.")
            else:
                output.put_text(f"No common words found for overall analysis in {group_name}.")
        except Exception as e:
            output.put_text(f"Error in overall NLTK analysis for {group_name}: {type(e).__name__} - {e}")
    else:
        output.put_text(f"No tokens found for overall analysis in {group_name}.")

    unique_emails_table = [[email] for email in set(local_emails_list)]
    unique_phones_table = [[ph] for ph in set(local_phones_list)]
    output.put_collapse('Found Emails and Numbers', [
        output.put_table(unique_emails_table, header=['Emails:']),
        output.put_table(unique_phones_table, header=['Numbers:'])
    ], open=True) # Open by default

    output.put_text(f"Total messages processed: {local_message_count}")
    # Caller should handle main app buttons like Close / Scroll Up.

if __name__ == '__main__':
    # This section is for potential direct execution or testing.
    # The primary way to use this module is by importing `perform_chat_analysis`.
    if len(sys.argv) > 1:
        json_file_arg = sys.argv[1]
        
        # For PyWebIO to work when script is run directly, it needs a server context.
        # The `perform_chat_analysis` function uses PyWebIO `output` calls,
        # which require an active PyWebIO session.
        # If `telanalysis.py` starts the PyWebIO server and then calls this function,
        # it will run within that server's context.
        # To run this file standalone for testing, you'd wrap it in a PyWebIO app:
        def standalone_app():
            config(theme='dark', title="Standalone Word Analysis Test")
            output.put_markdown("## Standalone Word Analysis Test")
            perform_chat_analysis(json_file_arg)

        # print(f"Starting standalone PyWebIO server for analyzing {json_file_arg}")
        # from pywebio import start_server
        # start_server(standalone_app, port=8081, debug=True)
        # Commented out start_server to prevent accidental execution during refactoring process.
        # The primary goal is importability.
        print(f"To test, uncomment the start_server call in __main__ and run with a JSON file argument.")
        print("Example: python words_analyze.py asset/your_chat.json")
        
    else:
        print("Usage (for standalone testing): python words_analyze.py <path_to_json_file>")

``````python
from utils import remove_chars_from_text, remove_emojis, clear_user, read_conf
import nltk_analyse
import sys
from pywebio import config, output, pin, session
import json, re, jmespath
from validate_email import validate_email
import phonenumbers
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Action map for different message types
ACTION_MAP = {
    'invite_members': 'Invite Member',
    'remove_members': 'Kicked Members',
    'join_group_by_link': 'Joined by Link',
    'pin_message': 'Pinned Message',
    # Add other actions as needed
}

# Initializing the sentiment analyzer
ANALYZER = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    try:
        score = ANALYZER.polarity_scores(str(text))
        return float(score['compound'])  # Converting to float for certainty
    except:
        return float(0.0)

def extract_emails_and_phone_numbers(text):
    """Extracts email addresses and phone numbers from a given text."""
    emails_list_local = []
    emails_re = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    for email_val in emails_re:
        if validate_email(email_val, verify=False):
            emails_list_local.append(email_val)
    phones_list_local = []
    phone_numbers_re = re.findall(r'\+?[0-9]{1,3}?[-. (]?[0-9]{1,4}[-. )]?[0-9]{1,4}[-. ]?[0-9]{1,9}', text)
    for phones_val in phone_numbers_re:
        try:
            phone_number = phonenumbers.parse(phones_val, None)
            if phonenumbers.is_valid_number(phone_number):
                phones_list_local.append(phones_val)
        except Exception:
            pass
    return emails_list_local, phones_list_local

def extract_text_from_message(message_obj):
    """Extracts text from a message object, handling nested structures."""
    texts_set = set()  # Using a set for unique text values

    if isinstance(message_obj, dict):
        if 'text' in message_obj:
            if isinstance(message_obj['text'], str) and message_obj['text'].strip():
                texts_set.add(message_obj['text'])
            elif isinstance(message_obj['text'], list):
                for item in message_obj['text']:
                    if isinstance(item, str): # Ensure item is a string
                        texts_set.add(item)
        
        if 'caption' in message_obj:
            if isinstance(message_obj['caption'], str) and message_obj['caption'].strip():
                texts_set.add(message_obj['caption'])
        
        entities = jmespath.search('text_entities[*].text', message_obj)
        if entities:
            for entity in entities:
                if isinstance(entity, str): # Ensure entity is a string
                    texts_set.add(entity)

        if 'forwarded_from' in message_obj: # Check if key exists
            texts_set.update(extract_text_from_message(message_obj['forwarded_from']))

        if 'reply_to_message' in message_obj: # Check if key exists
            texts_set.update(extract_text_from_message(message_obj['reply_to_message']))

        for key, value in message_obj.items():
            if isinstance(value, (list, dict)): 
                texts_set.update(extract_text_from_message(value))

    elif isinstance(message_obj, list):
        for item_in_list in message_obj:
            texts_set.update(extract_text_from_message(item_in_list))

    return texts_set

def process_message_data(message_item):
    """
    Processes a single message.
    Returns:
        tuple: (user_id_val, message_content, extracted_emails, extracted_phones, message_count_increment)
               message_content is (text, sentiment_score) or None for action-only messages.
    """
    user_id_val = jmespath.search('from_id', message_item)
    action_text_val = ""
    message_content = None # Will store (text, sentiment_score)
    message_count_increment = 0
    
    # Lists for this message's extracted data
    emails_this_message = []
    phones_this_message = []

    if not user_id_val:
        user_id_val = jmespath.search('actor_id', message_item)
        if user_id_val:
            user_id_val = user_id_val.replace(" ", "")
            action = jmespath.search('action', message_item)
            if action:
                tex = jmespath.search('text', message_item) or ''
                action_text_val = ACTION_MAP.get(action, action)
                
                if action in ['invite_members', 'remove_members']:
                    members = jmespath.search('members', message_item)
                    members_str = ",".join(str(x) for x in members if x)
                    message_content = (f"{action_text_val} - {members_str}", 0.0)
                else:
                    message_content = (f"{action_text_val} {tex}", 0.0)
                # Action messages don't count towards user text messages for NLTK analysis of user content
                # but are associated with the user.
                return user_id_val, message_content, emails_this_message, phones_this_message, 0 # No increment for "text" messages

    if not user_id_val: 
        return None, None, emails_this_message, phones_this_message, 0

    user_id_val = user_id_val.replace(" ", "")
    message_count_increment = 1 # Counts as one processed message from a user

    # Aggregate all unique texts from the message structure
    unique_texts = extract_text_from_message(message_item)
    
    # For simplicity, we'll take the first piece of text for sentiment, or join them.
    # NLTK analysis will later process all tokens from all texts.
    # This part focuses on associating a primary sentiment with the message event.
    # A more granular approach might create multiple (text, sentiment) tuples per message if multiple texts exist.
    
    primary_text_for_sentiment = " ".join(list(unique_texts)) if unique_texts else ""

    if primary_text_for_sentiment: # Ensure there is text
        sentiment_score = analyze_sentiment(primary_text_for_sentiment)
        message_content = (primary_text_for_sentiment, sentiment_score) # Store the aggregated text
        
        # Extract PII from all unique texts found in the message
        for text_segment in unique_texts:
            extracted_emails, extracted_phone_numbers = extract_emails_and_phone_numbers(text_segment)
            emails_this_message.extend(extracted_emails)
            phones_this_message.extend(extracted_phone_numbers)
            
    return user_id_val, message_content, emails_this_message, phones_this_message, message_count_increment


def perform_chat_analysis(json_file_path_param):
    """
    Performs chat analysis on a given JSON file.
    Handles PyWebIO outputs for displaying results.
    """
    local_emails_list, local_phones_list, local_all_tokens_list, local_users_data = [], [], [], {}
    local_message_count = 0

    output.put_html("<h1><center>Analyse of Telegram Chat<center></h1><br>")
    pin.put_input('ID_search_words_analyze', label='Search by User ID in results:')
    output.put_button("Search ID in Results", onclick=lambda: session.run_js(f'window.find(pin.pin.ID_search_words_analyze, true)'), color='warning')
    
    try:
        with open(json_file_path_param, 'r', encoding='utf-8', errors='replace') as data_file:
            data_content = json.load(data_file)
    except FileNotFoundError:
        output.put_error(f"Error: The file {json_file_path_param} was not found.")
        return
    except json.JSONDecodeError:
        output.put_error(f"Error: Could not decode JSON from {json_file_path_param}.")
        return


    sf_messages = jmespath.search('messages[*]', data_content)
    if not sf_messages:
        output.put_text("No messages found in the JSON file.")
        return
        
    group_name = jmespath.search('name', data_content) or "Chat" 

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_msg = {executor.submit(process_message_data, msg_item): msg_item for msg_item in sf_messages}

        for future in as_completed(future_to_msg):
            try:
                user_id, content, emails_from_msg, phones_from_msg, msg_increment = future.result()
                
                if user_id and content: # Ensure user_id and content are not None
                    if user_id not in local_users_data:
                        local_users_data[user_id] = []
                    local_users_data[user_id].append(content) # content is (text, sentiment_score)
                
                local_emails_list.extend(emails_from_msg)
                local_phones_list.extend(phones_from_msg)
                local_message_count += msg_increment
            except Exception as e:
                msg = future_to_msg[future]
                msg_id_for_error = msg.get('id', 'N/A')
                output.put_text(f"Error processing message ID {msg_id_for_error}: {type(e).__name__} - {e}")
    
    # Make PII lists unique
    local_emails_list = list(set(local_emails_list))
    local_phones_list = list(set(local_phones_list))

    for user_id_key, user_messages_list in local_users_data.items():
        user_from_name = ""
        # Find user's display name from the original messages list
        for m_item_orig in sf_messages: 
            if jmespath.search('from_id', m_item_orig) == user_id_key:
                user_from_name = jmespath.search('from', m_item_orig)
                break
            elif jmespath.search('actor_id', m_item_orig) == user_id_key: 
                user_from_name = jmespath.search('actor', m_item_orig) 
                if not user_from_name: user_from_name = user_id_key # Default to ID if 'actor' field is empty
                break
        
        user_display_name = f"{user_from_name} ({user_id_key})" if user_from_name and user_from_name != user_id_key else user_id_key
            
        user_sentiment_scores = [float(x[1]) for x in user_messages_list if x and isinstance(x[1], (float, int))]
        average_user_sentiment = sum(user_sentiment_scores) / len(user_sentiment_scores) if user_sentiment_scores else 0.0
        
        try:
            most_com_val = read_conf('most_com')
            # Extract texts for NLTK: user_messages_list contains (text, score)
            message_texts_for_nltk = [str(x[0]) for x in user_messages_list if x and x[0]] 
            
            gemy_table_data = []
            if message_texts_for_nltk:
                genuy_freq_dist, tokens_list = nltk_analyse.analyse(message_texts_for_nltk, most_com_val)
                gemy_table_data = [[word, count] for word, count in genuy_freq_dist]
                local_all_tokens_list.extend(tokens_list)

            if user_messages_list or gemy_table_data: 
                output.put_collapse(user_display_name, [
                    output.put_text(f'Messages of {user_display_name} (Count: {len(user_messages_list)})'),
                    output.put_text(f'Average Sentiment Score (Compound) for {user_display_name}: {average_user_sentiment:.2f}'),
                    output.put_table([[x[0], f"{x[1]:.2f}"] for x in user_messages_list if x], header=['Message', 'Sentiment Score (Compound)']),
                    output.put_table(gemy_table_data, header=['Word', 'Count'])
                ], open=False)

        except Exception as ex:
            output.put_text(f"Error during NLTK/display for user {user_display_name}: {type(ex).__name__} - {ex}")

    most_com_overall = read_conf('most_com')
    if local_all_tokens_list:
        try:
            fdist_all, _ = nltk_analyse.analyse_all(local_all_tokens_list, most_com_overall) # data_words_all not used later
            
            if fdist_all:
                all_chat_table_data = [[word, count] for word, count in fdist_all]
                output.put_collapse(f'TOP words of {group_name} (Total analyzed tokens: {len(local_all_tokens_list)})', [
                    output.put_table(all_chat_table_data, header=['Word', 'Count']),
                ], open=True)

                all_user_average_sentiments = []
                for _, user_msgs_list_item in local_users_data.items(): # Iterate through values of local_users_data
                    user_sent_scores = [float(x[1]) for x in user_msgs_list_item if x and isinstance(x[1], (float, int))]
                    if user_sent_scores:
                        all_user_average_sentiments.append(sum(user_sent_scores) / len(user_sent_scores))
                
                if all_user_average_sentiments:
                    average_chat_sentiment = sum(all_user_average_sentiments) / len(all_user_average_sentiments)
                    output.put_text(f'Average Chat Sentiment Score (Compound) for {group_name}: {average_chat_sentiment:.2f}')
                else:
                    output.put_text(f"Overall sentiment score for {group_name} could not be calculated (no user sentiments).") # Minor rephrase for consistency
            else:
                output.put_text(f"No common words found for overall analysis in {group_name}.")
        except Exception e:
            output.put_text(f"Error in overall NLTK analysis for {group_name}: {type(e).__name__} - {e}")
    else:
        output.put_text(f"No tokens found for overall analysis in {group_name}.")

    unique_emails_table = [[email_addr] for email_addr in local_emails_list] # Already unique due to set conversion earlier
    unique_phones_table = [[phone_num] for phone_num in local_phones_list] # Already unique
    output.put_collapse('Found Emails and Numbers', [
        output.put_table(unique_emails_table, header=['Emails:']),
        output.put_table(unique_phones_table, header=['Numbers:'])
    ], open=True)

    output.put_text(f"Total messages processed (where text or action was identified with a user): {local_message_count}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_file_arg = sys.argv[1]
        
        # This setup is for standalone testing of words_analyze.py with PyWebIO
        def standalone_app():
            # Basic PyWebIO config for the standalone app
            config(theme='dark', title="Standalone Word Analysis Test")
            output.put_markdown("## Standalone Word Analysis Test")
            # Global buttons for a standalone app context
            output.put_button("Scroll Down", onclick=lambda: session.run_js('window.scrollTo(0, document.body.scrollHeight)'))
            output.put_button("Close (Standalone Test)", onclick=lambda: session.run_js('window.close()'), color='danger')
            
            perform_chat_analysis(json_file_arg)

            output.put_button("Scroll Up (Standalone Test)", onclick=lambda: session.run_js('window.scrollTo(document.body.scrollHeight, 0)'))

        # To run this:
        # 1. Ensure you have a sample JSON file (e.g., asset/your_chat.json)
        # 2. Uncomment the line below.
        # 3. Run from terminal: python words_analyze.py asset/your_chat.json
        # from pywebio import start_server
        # start_server(standalone_app, port=8081, debug=True) 
        print(f"To test this script standalone with PyWebIO, uncomment the start_server call in the __main__ block.")
        print(f"Then run: python words_analyze.py <path_to_your_json_file>")
    else:
        print("Usage (for standalone testing): python words_analyze.py <path_to_json_file>")
```
