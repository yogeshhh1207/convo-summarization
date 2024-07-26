import os
import subprocess
from googletrans import Translator
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import torch
import numpy as np
import pathlib

def translate_text(text, src_lang='hi', dest_lang='en'):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text

def generate_key_points(text):
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = "extract key points: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    key_points_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    key_points = tokenizer.decode(key_points_ids[0], skip_special_tokens=True)
    return key_points

def summarize_text(text):
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=device)
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    return summary

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def perform_ner(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def detect_non_compliance(conversation):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', num_labels=2)
    
    non_compliance_sentences = []
    sentences = conversation.split('.')
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1).numpy()
        
        if np.argmax(predictions) == 1:  
            non_compliance_sentences.append(sentence)
    
    return non_compliance_sentences

def main(input_file, output_file, key_points_file, summary_file, sentiment_file, ner_file, non_compliance_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        conversation_hindi = file.read()

    conversation_english = translate_text(conversation_hindi)

    speaker1_lines = []
    speaker2_lines = []
    for line in conversation_english.split('\n'):
        if line.startswith('Speaker 1:'):
            speaker1_lines.append(line.replace('Speaker 1:', ''))
        elif line.startswith('Speaker 2:'):
            speaker2_lines.append(line.replace('Speaker 2:', ''))


    speaker1_text = ' '.join(speaker1_lines)
    speaker2_text = ' '.join(speaker2_lines)


    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(conversation_english)


    key_points = generate_key_points(conversation_english)
    with open(key_points_file, 'w', encoding='utf-8') as file:
        file.write(key_points)


    summary = summarize_text(conversation_english)
    with open(summary_file, 'w', encoding='utf-8') as file:
        file.write(summary)


    sentiment_speaker1 = analyze_sentiment(speaker1_text)
    sentiment_speaker2 = analyze_sentiment(speaker2_text)
    with open(sentiment_file, 'w', encoding='utf-8') as file:
        file.write(f"Speaker 1 Sentiment: {sentiment_speaker1}\n")
        file.write(f"Speaker 2 Sentiment: {sentiment_speaker2}\n")


    ner_speaker1 = perform_ner(speaker1_text)
    ner_speaker2 = perform_ner(speaker2_text)
    with open(ner_file, 'w', encoding='utf-8') as file:
        file.write(f"Speaker 1 Entities: {ner_speaker1}\n")
        file.write(f"Speaker 2 Entities: {ner_speaker2}\n")


    non_compliance_sentences = detect_non_compliance(conversation_english)
    with open(non_compliance_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(non_compliance_sentences))

if __name__ == "__main__":
    input_file = "conversation_hindi.txt"
    output_file = "conversation_english.txt"
    key_points_file = "key_points.txt"
    summary_file = "summary.txt"
    sentiment_file = "sentiment.txt"
    ner_file = "ner.txt"
    non_compliance_file = "non_compliance.txt"
    main(input_file, output_file, key_points_file, summary_file, sentiment_file, ner_file, non_compliance_file)