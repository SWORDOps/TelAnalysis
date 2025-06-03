import json
import jmespath
import nltk_analyse
import utils
import time
from pywebio import input, config
from pywebio.output import put_html, put_text, put_image, put_table
from wordcloud import WordCloud


# Reading the configuration
select_type_stem = utils.read_conf('select_type_stem')
most_com = utils.read_conf('most_com_channel')

def channel(filename):
    # Extracting the channel name from the file
    filename = filename.split(".")[0].split("/")[1]
    with open(f'asset/{filename}.json', 'r', encoding='utf-8') as f:
        jsondata = json.load(f)
        name_channel = jmespath.search('name', jsondata)

        # Displaying the channel name
        put_html(f"<center><h1>{name_channel}</h1></center>")
        messages_find = jmespath.search('messages[*].text', jsondata)

        text_list = []
        
        # Processing channel messages
        for message in messages_find:
            if isinstance(message, list):
                for mes in message:
                    text = jmespath.search('text', mes) or mes
                    text_list.append(utils.remove_emojis(text))
            else:
                message = message.replace("   ", " ").replace("\n", "").replace("\t", "").strip()
                if len(message) > 4:
                    text_list.append(utils.remove_emojis(message))

        # Text analysis and word cloud generation
        fdist, tokens = nltk_analyse.analyse(text_list, most_com)
        all_tokens = list(tokens)
        all_tokens, data = nltk_analyse.analyse_all(all_tokens, most_com)
        
        # Word cloud generation
        text_raw = " ".join(data)
        wordcloud = WordCloud().generate(text_raw)
        filename_path = f'asset/{filename}_wordcloud.png'
        wordcloud.to_file(filename_path)


        # Displaying the result
        with open(filename_path, 'rb') as img_file:
            img = img_file.read()
        
        time.sleep(2)
        put_text(f"Wordcloud[{most_com}]:")
        put_image(img, width='600px')
        put_text(f"\nCount of all tokens: {len(tokens)}")
        put_text(f"\nСhannel frequency analysis[{most_com}]:")

        # Formatting data for the table
        gemy = [[x, y] for x, y in all_tokens]
        put_table(gemy, header=['word', 'count'])
