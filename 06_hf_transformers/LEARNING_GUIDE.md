# Hugging Face Transformers - Learning Guide

## 🎯 Module Overview

Master the Hugging Face ecosystem and modern transformer architectures. Learn to use, fine-tune, and deploy state-of-the-art NLP and multimodal models.

## 📚 What You'll Learn

- Transformer architecture fundamentals
- Using pre-trained models from Hugging Face Hub
- Fine-tuning models on custom datasets
- Pipeline API for quick inference
- Model evaluation and benchmarking
- Saving and sharing models

## 🎓 Learning Objectives

- [ ] Load pre-trained transformers
- [ ] Use pipeline API for common tasks
- [ ] Prepare datasets for fine-tuning
- [ ] Fine-tune BERT for classification
- [ ] Evaluate model performance
- [ ] Push models to Hugging Face Hub

## 📝 Key Concepts

### Transformer Architecture
- **Self-Attention**: Contextual word relationships
- **Encoder-only**: BERT (classification, NER)
- **Decoder-only**: GPT (generation)
- **Encoder-Decoder**: T5, BART (translation, summarization)

### Pre-training vs Fine-tuning
- **Pre-training**: Learn language on massive datasets (millions)
- **Fine-tuning**: Adapt to specific task (hundreds/thousands)

### Common Models
- **BERT**: Classification, NER, Q&A
- **GPT-2/3**: Text generation
- **T5**: Text-to-text (any task)
- **BART**: Summarization, generation
- **RoBERTa**: Improved BERT

## 🚀 Exercises & Tasks

### Task 1: Model Inference
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this audiobook!")
```

### Task 2: Load and Explore Model
- Load BERT from Hub
- Inspect architecture
- Check number of parameters
- Test on sample inputs

### Task 3: Prepare Custom Dataset
- Format data for classification
- Create train/val/test splits
- Tokenize with model tokenizer
- Create DataLoader

### Task 4: Fine-tune BERT
- Choose appropriate task
- Set hyperparameters
- Train for several epochs
- Monitor loss and accuracy

### Task 5: Evaluate and Save
- Compute accuracy, F1, precision, recall
- Plot training curves
- Save fine-tuned model
- Test on new examples

## 📊 Success Criteria

- ✅ Use any HF model from the Hub
- ✅ Fine-tune on custom data
- ✅ Achieve >85% accuracy on task
- ✅ Understand transformer architecture
- ✅ Save and reload models

## 🔧 Required Libraries

```bash
pip install transformers
pip install datasets
pip install accelerate
pip install evaluate
```

## 💡 Model Selection Guide

| Task | Recommended Model |
|------|-------------------|
| Classification | BERT, RoBERTa, DistilBERT |
| Text Generation | GPT-2, GPT-Neo |
| Summarization | BART, T5, Pegasus |
| Translation | MarianMT, mBART, T5 |
| Q&A | BERT, RoBERTa, ELECTRA |

## 🔗 Next Steps

→ **[07_data_preparation](../07_data_preparation/)** to build quality datasets

**Time Estimate**: 8-12 hours
