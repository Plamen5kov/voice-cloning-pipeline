# Natural Language Processing - Learning Guide

## 🎯 Module Overview

Learn to process, analyze, and understand text data using modern NLP techniques. Essential for preparing text inputs for TTS systems and audiobook generation.

## 📚 What You'll Learn

- Text tokenization and preprocessing
- Sentence and paragraph segmentation
- Named entity recognition (NER)
- Sentiment analysis
- Text summarization
- Language detection

## 🎓 Learning Objectives

- [ ] Tokenize text with Hugging Face tokenizers
- [ ] Extract named entities (people, places, organizations)
- [ ] Analyze sentiment of text passages
- [ ] Summarize long documents
- [ ] Detect and handle multiple languages
- [ ] Clean and normalize text for TTS

## 📝 Key Concepts

### Tokenization
- **Word-level**: Split on spaces/punctuation
- **Subword**: BPE, WordPiece (used in BERT)
- **Character-level**: For rare words

### Named Entity Recognition (NER)
- **PER**: Person names
- **LOC**: Locations
- **ORG**: Organizations
- **MISC**: Miscellaneous entities

### Sentiment Analysis
- **Positive/Negative**: Binary classification
- **Fine-grained**: 1-5 star ratings
- **Aspect-based**: Sentiment per topic

## 🚀 Exercises & Tasks

### Task 1: Text Tokenization
- Load pre-trained tokenizer
- Tokenize sentences and paragraphs
- Handle special characters
- Understand token IDs vs text

### Task 2: Named Entity Extraction
- Use spaCy or Hugging Face NER
- Extract all person names from a chapter
- Create character list for audiobook

### Task 3: Sentiment Analysis
- Analyze sentiment per chapter
- Identify emotional arcs
- Adjust TTS parameters based on sentiment

### Task 4: Text Summarization
- Use T5 or BART for summarization
- Generate chapter summaries
- Create audiobook descriptions

### Task 5: Dialogue Detection
- Extract all quoted speech
- Identify speakers
- Tag dialogue vs narration

## 📊 Success Criteria

- ✅ Tokenize text for ML models
- ✅ Extract entities and relationships
- ✅ Analyze sentiment accurately
- ✅ Generate quality summaries
- ✅ Prepare clean text for TTS

## 🔧 Required Libraries

```bash
pip install transformers
pip install spacy
pip install nltk
python -m spacy download en_core_web_sm
```

## 🔗 Next Steps

→ **[06_hf_transformers](../06_hf_transformers/)** for advanced transformer models

**Time Estimate**: 6-8 hours
