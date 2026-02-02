# Python Programming - Learning Guide

## 🎯 Module Overview

Master essential Python programming skills for AI and data science work. This module focuses on practical skills you'll use daily: file I/O, text processing, data manipulation, and automation.

## 📚 What You'll Learn

- File reading and writing operations
- String manipulation and text processing
- Working with structured data
- Audio file format conversion
- Scripting and automation
- Python best practices for ML projects

## 🎓 Learning Objectives

By completing this module, you should be able to:
- [ ] Read and write text files programmatically
- [ ] Parse and extract information from documents
- [ ] Count words, sentences, and analyze text structure
- [ ] Convert between audio formats (M4A to WAV)
- [ ] Organize and process datasets with scripts
- [ ] Write reusable, modular code

## 📝 Key Concepts

### File I/O Operations
**Why**: All ML projects involve reading data and saving results
**Skills**: open(), read(), write(), with statements, path handling

### Text Processing
**Why**: NLP and TTS require extensive text manipulation
**Skills**: String methods, regex, parsing, cleaning

### Audio Processing Basics
**Why**: Voice cloning requires working with audio files
**Skills**: Format conversion, metadata handling, batch processing

## 🚀 Exercises & Tasks

### Task 1: Read Text File (`read_text_file.py`)
Write a script to:
- Open a text file
- Read and print its contents
- Handle file not found errors

**Learning Point**: Always use `with` statement for automatic file closing

**Expected Output**:
```
File contents displayed line by line
```

### Task 2: Count Words & Sentences (`count_words_sentences.py`)
Create a script that:
- Reads a text file
- Counts total words
- Counts total sentences
- Reports statistics

**Learning Point**: Understanding text structure is crucial for NLP

**Expected Output**:
```
Total words: 1234
Total sentences: 56
Average words per sentence: 22.0
```

### Task 3: Audio Format Conversion (`convert_m4a_to_wav.py`)
Build a script to:
- Convert M4A files to WAV format
- Handle multiple files in a directory
- Preserve audio quality
- Use appropriate sampling rate (24kHz for TTS)

**Learning Point**: WAV format is standard for ML audio processing

**Expected Output**:
```
Converted: voice_sample1.m4a → voice_sample1.wav
Converted: voice_sample2.m4a → voice_sample2.wav
```

### Task 4: Extract Dialogue Lines
Write a script that:
- Identifies quoted dialogue in text
- Extracts all dialogue lines
- Saves to a separate file

**Learning Point**: Useful for creating audiobook datasets

**Expected Output**:
```
"Hello, how are you?"
"I'm doing well, thanks!"
```

### Task 5: Chapter Splitting
Create a script to:
- Detect chapter headings (e.g., "Chapter 1", "CHAPTER ONE")
- Split book into separate chapter files
- Number files sequentially

**Learning Point**: Organizing data properly speeds up training

## 📊 Success Criteria

You've completed this module when you can:
- ✅ Read and write files without errors
- ✅ Parse and analyze text structure
- ✅ Convert audio files between formats
- ✅ Batch process multiple files
- ✅ Handle errors gracefully
- ✅ Write clean, documented code

## 🔧 Required Libraries

```bash
pip install pydub
# For audio conversion, also install ffmpeg:
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

## 🔗 Next Steps

Once you're comfortable with Python:
→ Move to **[02_deep_learning_basics](../02_deep_learning_basics/)** to start ML training

## 💡 Best Practices

1. **Use meaningful variable names**: `word_count` not `wc`
2. **Write functions**: Modular code is reusable code
3. **Handle errors**: Use try/except for file operations
4. **Document your code**: Comments explain why, not what
5. **Test with small files first**: Debug faster with minimal data

## 📖 Additional Resources

- [Python File I/O](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [pydub Documentation](https://github.com/jiaaro/pydub)
- [Python Regular Expressions](https://docs.python.org/3/library/re.html)

## ⚠️ Common Pitfalls

**Pitfall**: Forgetting to close files
**Solution**: Always use `with open()` context manager

**Pitfall**: Hardcoded file paths
**Solution**: Use relative paths or command-line arguments

**Pitfall**: Not handling encoding
**Solution**: Specify encoding='utf-8' when opening text files

---

**Time Estimate**: 4-6 hours (practice is key!)
