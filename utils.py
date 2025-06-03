import re
import string
import emoji
import os
import subprocess
import platform
import json, jmespath, requests
from pywebio.output import put_html,toast,put_text,put_image,put_collapse, put_button, put_code, clear, put_file, popup, put_table

# spec_chars is used by remove_chars_from_text and remove_emojis
spec_chars = string.punctuation + '\n\xa0«»\t—…"<>?!.,;:꧁@#$%^&*()_-+=№%༺༺\༺/༺•'

##config telanalysis
def read_conf(option):
    try:
        with open('config.json', 'r', encoding='utf-8') as read_conf_file: # Renamed variable to avoid conflict
            config_data = json.load(read_conf_file) # Renamed variable
            selected_option = jmespath.search(f'{option}', config_data) # Renamed variable
        return selected_option
    except FileNotFoundError:
        # If config.json doesn't exist, create it with default values
        default_config = {"select_type_stem": "Off", "most_com": 30, "most_com_channel":100}
        write_conf(default_config)
        return default_config.get(option)
    except Exception as e:
        # For other exceptions, print an error and return a default value or None
        print(f"Error reading config: {e}")
        # Fallback to creating default config if read fails for other reasons too
        default_config = {"select_type_stem": "Off", "most_com": 30, "most_com_channel":100}
        write_conf(default_config)
        return default_config.get(option)


def write_conf(dct):
    with open('config.json', 'w', encoding='utf-8') as fw: # Added encoding
        json.dump(dct, fw, indent=4) # Added indent for readability

def clear_console():
    system = platform.system()
    if system == 'Windows':
        subprocess.run('cls', shell=True, check=False) # Added check=False for robustness
    elif system == 'Darwin' or system == 'Linux':
        subprocess.run('clear', shell=True, check=False) # Added check=False
        
def open_url():
    system = platform.system()
    url_to_open = 'http://127.0.0.1:9993' # Defined URL
    if system == 'Windows':
        subprocess.run(f'start {url_to_open}', shell=True, check=False)
    elif system == 'Darwin': # macOS uses 'open'
        subprocess.run(['open', url_to_open], check=False)
    elif system == 'Linux':
        subprocess.run(['xdg-open', url_to_open], check=False) # xdg-open is common on Linux

def remove_chars_from_text(text, char_list=None): # Renamed 'char' to 'char_list' for clarity
    if text is None:
        return ""
    text_str = str(text) # Ensure input is string
    if char_list is None:
        char_list = spec_chars
    
    pattern = f"[{re.escape(char_list)}]"
    cleaned_text = re.sub(pattern, ' ', text_str)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def fetch_osint_data(tg_id_input, api_token):
    """
    EXPERIMENTAL: Fetches user data from osintframework.ru.
    This feature is experimental, possibly incomplete, and requires a valid API token.
    It is not currently used by the main application.
    The UI interactions (toast/popup) should ideally be handled by the calling code
    by having this function return data instead of directly creating UI elements.
    """
    toast(content='Fetching OSINT data... Please wait.', duration=0) # Moved toast here

    # Validate or sanitize tg_id_input before using it
    if not isinstance(tg_id_input, (str, int)):
        toast("Invalid Telegram ID format.", color='error', duration=3)
        return None # Or raise an error

    try:
        # Assuming tg_id_input might come with "user" or "channel" prefix
        tg_id_str = str(tg_id_input).lower().replace("user", "").replace("channel", "").strip()
        if not tg_id_str.isdigit(): # Basic check if it's a number after stripping prefixes
            toast(f"Invalid Telegram ID: {tg_id_input}. Must be a numeric ID.", color='error', duration=3)
            return None
        
        tg_id_numeric = int(tg_id_str)
    except ValueError:
        toast(f"Invalid Telegram ID format after cleaning: {tg_id_str}", color='error', duration=3)
        return None

    if not api_token:
        toast("API token for osintframework.ru is missing.", color='error', duration=3)
        # In a real scenario, you might raise an error or handle this more gracefully.
        # For now, just returning as it's experimental.
        return None

    # Initialize lists for collecting data
    phonenumbers_list = []
    ids_list = []
    telegram_ids_list = []
    firstnames_list = []
    surnames_list = []
    emails_list = []
    trades_list = []
    social_medias_list = []
    addresses_list = []
    technicals_data_list = []

    try:
        # print(f"Fetching data for Telegram ID: {tg_id_numeric}")
        req = requests.post(
            'https_url_placeholder/api/telegram/telegram-user-somevendor', # Placeholder URL, actual was osintframework.ru
            json={"telegram_id": tg_id_numeric},
            headers={"Authorization": api_token}, # Use the passed token
            timeout=60
        )
        req.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        response_json = req.json()
        
        # The jmespath expression for finded_data was: "telegram_id_somevendor.finded_data"
        # This structure might be specific to the API's response.
        finded_data = jmespath.search("telegram_id_somevendor.finded_data", response_json)

        if finded_data is None: # Check if the path itself is missing or returns None
            toast("OSINT data structure in response is not as expected or 'finded_data' is null.", color='warn', duration=3)
            # print(f"Full response for debugging: {response_json}")
            return None 
        if not finded_data: # Empty list
            toast("No OSINT data found for this ID.", duration=2)
            return None

        for item_dict in finded_data: # finded_data is a list of dictionaries
            if not isinstance(item_dict, dict): continue # Skip if item is not a dict

            for key, value in item_dict.items():
                if value is None: continue # Skip None values to simplify appends

                # Using .get(key) or checking key in item_dict is safer if keys are not guaranteed
                if 'phone_number' in key: phonenumbers_list.append(str(value))
                elif 'id' in key: ids_list.append(str(value)) # Assuming general ID
                elif 'telegram_id' in key: telegram_ids_list.append(str(value)) # Specific Telegram ID
                elif 'firstname' in key: firstnames_list.append(str(value))
                elif 'surname' in key: surnames_list.append(str(value))
                elif 'email' in key: emails_list.append(str(value))
                elif 'trade' in key: trades_list.append(str(value))
                elif 'social_media' in key: social_medias_list.append(str(value))
                elif 'address' in key: addresses_list.append(str(value))
                elif 'technical_data' in key: technicals_data_list.append(str(value))
        
        # Remove duplicates by converting to set and back to list
        collected_data = {
            "PhoneNumbers": list(set(phonenumbers_list)),
            "TelegramIDs": list(set(telegram_ids_list)),
            "OtherIDs": list(set(ids_list)), # General IDs
            "FirstNames": list(set(firstnames_list)),
            "Surnames": list(set(surnames_list)),
            "Emails": list(set(emails_list)),
            "Addresses": list(set(addresses_list)),
            "Trades": list(set(trades_list)),
            "SocialMedia": list(set(social_medias_list)),
            "TechnicalData": list(set(technicals_data_list))
        }

        # For PyWebIO popup, prepare data for table
        # This part is still UI-related and ideally should be handled by the caller.
        # For now, it's isolated here.
        if any(collected_data.values()): # If any data was actually collected
            popup_table_data = [
                collected_data["PhoneNumbers"],
                collected_data["TelegramIDs"],
                collected_data["FirstNames"], # Surnames were separate, combining for simplicity or add column
                collected_data["Emails"],
                collected_data["Addresses"],
                collected_data["Trades"],
                collected_data["SocialMedia"],
                collected_data["TechnicalData"]
            ]
            # Ensure all lists in popup_table_data are of the same primary type for table display (e.g. strings)
            # Or handle complex structures by formatting them into strings.
            # For now, assuming they are mostly lists of strings.
            
            # Transpose or format for better table view if multiple entries per category
            # The original code put lists directly into cells. Let's make it more readable.
            display_rows = []
            max_rows = max(len(v) for v in collected_data.values() if v) if any(collected_data.values()) else 0

            for i in range(max_rows):
                row = [
                    collected_data["PhoneNumbers"][i] if i < len(collected_data["PhoneNumbers"]) else "",
                    collected_data["TelegramIDs"][i] if i < len(collected_data["TelegramIDs"]) else "",
                    (collected_data["FirstNames"][i] if i < len(collected_data["FirstNames"]) else "") + " " + (collected_data["Surnames"][i] if i < len(collected_data["Surnames"]) else ""),
                    collected_data["Emails"][i] if i < len(collected_data["Emails"]) else "",
                    collected_data["Addresses"][i] if i < len(collected_data["Addresses"]) else "",
                    collected_data["Trades"][i] if i < len(collected_data["Trades"]) else "",
                    collected_data["SocialMedia"][i] if i < len(collected_data["SocialMedia"]) else "",
                    collected_data["TechnicalData"][i] if i < len(collected_data["TechnicalData"]) else ""
                ]
                display_rows.append(row)


            popup_title = f'OSINT Info for Telegram ID: {tg_id_numeric}'
            if not display_rows: # If after processing, there's nothing to show (e.g. all lists were empty)
                 put_text("No structured data to display in table, but some information might have been found.")
            else:
                popup(popup_title, [
                    put_table(display_rows, header=['Phone', 'Telegram ID', 'Name', 'Email', 'Address', 'Trades', 'Social Media', 'Tech Data'])
                ])
            return collected_data # Return the structured data
        else:
            toast("No specific OSINT details found after parsing.", duration=2)
            return None


    except requests.exceptions.HTTPError as http_err:
        toast(f"OSINT API HTTP error: {http_err}", color='error', duration=5)
        print(f"HTTP error: {http_err} - Response: {req.text if 'req' in locals() else 'N/A'}")
    except requests.exceptions.RequestException as req_err:
        toast(f"OSINT API request error: {req_err}", color='error', duration=5)
        print(f"Request error: {req_err}")
    except json.JSONDecodeError:
        toast("Error decoding OSINT API response.", color='error', duration=3)
        print(f"JSONDecodeError - Response: {req.text if 'req' in locals() else 'N/A'}")
    except Exception as e:
        toast(f"An unexpected error occurred during OSINT fetch: {type(e).__name__}", color='error', duration=3)
        print(f"Unexpected error in fetch_osint_data: {e}")
    
    return None # Return None in case of error or no data


def remove_emojis(data_input):
    """
    Removes emojis from a string using the 'emoji' library.
    Also performs additional cleaning like removing special characters and ASCII characters.
    """
    if data_input is None:
        return ""
    text = str(data_input)

    # Remove emojis using the emoji library
    text = emoji.replace_emoji(text, replace='')

    # Remove special characters (using the global spec_chars)
    text = remove_chars_from_text(text, spec_chars) # spec_chars is global

    # Remove ASCII characters - this was part of the original function.
    text = re.sub(r'[\x00-\x7f]', ' ', text) 

    # Standard space cleanup
    text = text.replace("  ", " ").strip()
    
    return text

def clear_user(user):
    # Removing special characters, emojis, and cleaning the text (for user identifiers)
    user_text = str(user)
    # Initial specific character removal for user names/IDs
    user_text = user_text.replace(" ", "").replace('"', '').replace(".", "").replace("꧁", "")
    
    # General character removal (punctuation, etc.)
    user_text = remove_chars_from_text(user_text) # Uses global spec_chars
    
    # Emoji removal (which now also includes some char removal and ASCII filtering)
    user_text = remove_emojis(user_text) 
    
    # Final strip
    return user_text.strip()