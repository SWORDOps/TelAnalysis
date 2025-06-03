# TelAnalysis
<video src="https://user-images.githubusercontent.com/107117398/223553362-8f24206e-55de-4ed2-bdd1-1d6093f8d60b.mp4"></video>
![image](https://user-images.githubusercontent.com/107117398/223553327-a0ef0115-6cfe-4c38-9f0b-67062354a79c.png)
![image](https://user-images.githubusercontent.com/107117398/223553309-ba92ee44-ff54-4e3e-b49a-70596cde4198.png)
![image](https://user-images.githubusercontent.com/107117398/223553300-a5874615-fe67-4f8d-a042-df3aa5e3b0e6.png)
![image](https://user-images.githubusercontent.com/107117398/209858730-fe6ff0a3-9fcd-4d13-be6a-3f2a6bdd198b.png)

# TelAnalysis

## Description

TelAnalysis is a tool for analyzing messages in Telegram chats, groups, and channels. It helps extract text, identify keywords, perform sentiment analysis (calculating a compound positive/negative/neutral score for messages), and extract contact information such as email addresses and phone numbers.

## New Features and Improvements

1.  **Sentiment Analysis**:
    *   Performs sentiment analysis for each message, displaying a compound score (ranging from -1, most negative, to +1, most positive).
    *   Calculates and displays the average sentiment score for each user.
    *   Provides an overall average sentiment score for all messages in the analyzed chat or channel.

2.  **Improved Message Processing**:
    *   Fixed errors in calculating average sentiment scores to avoid exceptions.
    *   Added data type checks for more robust processing.

3.  **Contact Information Extraction**:
    *   Added functionality to extract email addresses and phone numbers from messages.

4.  **Code Refactoring & Stability**:
    *   Significant portions of the codebase have been refactored for improved clarity, maintainability, and stability.
    *   Enhanced error handling for NLTK resource downloads and configuration file management.
    *   Improved UI flow and user feedback in the web interface.

## Interpreting Sentiment Scores

The sentiment analysis uses VaderSentiment, which provides a compound score:
*   **Positive sentiment:** compound score >= 0.05
*   **Neutral sentiment:** compound score > -0.05 and < 0.05
*   **Negative sentiment:** compound score <= -0.05

The compound score is a normalized, weighted composite score ranging from -1 (most extreme negative) to +1 (most extreme positive).

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/krakodjaba/TelAnalysis.git
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you have Python 3.x installed.)

## Usage

Run the script from the project's root directory:

```bash
python3 telanalysis.py
```

The application will attempt to open in your default web browser at `http://127.0.0.1:9993`. If it doesn't open automatically, please navigate to this URL.

## Contributing

If you want to contribute, please fork the repository and submit a pull request.

## Donations

If you liked the project and want to support its development, you can donate for a coffee! ☕️
tg@e_isaevsan
```
      )  (
     (   ) )
      ) ( (
 mrf_______)_
 .-'---------|  
( C|/\/\/\/\/|
 '-./\/\/\/\/|
   '_________'
    '-------'
```

Thank you for your support!
