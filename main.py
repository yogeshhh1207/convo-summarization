import os
from googletrans import Translator
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to translate text from Hindi to English
def translate_text(text, src_lang='hi', dest_lang='en'):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text

# Function to generate key points using T5 model
def generate_key_points(text):
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = "extract key points: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    key_points_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    key_points = tokenizer.decode(key_points_ids[0], skip_special_tokens=True)
    return key_points

# Function to summarize the conversation using T5 model
def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    return summary

# Function to perform sentiment analysis using VADER
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def main(input_file, output_file, key_points_file, summary_file, sentiment_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        conversation_hindi = file.read()

    # Translate the conversation to English
    conversation_english = translate_text(conversation_hindi)

    # Extract lines spoken by the Recovery Agent (RA) and Borrower (B)
    ra_lines = []
    b_lines = []
    for line in conversation_english.split('\n'):
        if line.startswith('RA:'):
            ra_lines.append(line)
        elif line.startswith('B:'):
            b_lines.append(line)

    # Join lines into single text
    ra_text = ' '.join(ra_lines)
    b_text = ' '.join(b_lines)

    # Write the translated conversation to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(conversation_english)

    # Generate the key points for the Recovery Agent
    key_points = generate_key_points(conversation_english)
    with open(key_points_file, 'w', encoding='utf-8') as file:
        file.write(key_points)

    # Summarize the conversation
    summary = summarize_text(conversation_english)
    with open(summary_file, 'w', encoding='utf-8') as file:
        file.write(summary)

    # Perform sentiment analysis
    ra_sentiment = analyze_sentiment(ra_text)
    b_sentiment = analyze_sentiment(b_text)

    # Consolidate sentiment scores
    sentiment_results = f"Agent Sentiment: {ra_sentiment}\nBorrower Sentiment: {b_sentiment}"
    with open(sentiment_file, 'w', encoding='utf-8') as file:
        file.write(sentiment_results)

    print("Translation, key points extraction, summarization, and sentiment analysis completed successfully.")

if __name__ == "__main__":
    input_file = 'conversation_hindi.txt'
    output_file = 'conversation_english.txt'
    key_points_file = 'key_points.txt'
    summary_file = 'summary.txt'
    sentiment_file = 'sentiment.txt'

    main(input_file, output_file, key_points_file, summary_file, sentiment_file)
