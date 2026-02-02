# Voice Cloning Pipeline 🎙️

A hands-on learning path for building a modern voice cloning pipeline using deep learning, TTS systems, and related open-source tools. This project is designed for learning, experimentation, and developing AI engineering skills through practical implementation.

---

## 📚 Overview of Learning Materials

This repository contains a **structured learning path** with 13 modules covering everything from Python basics to deploying production ML systems. Each module has its own **LEARNING_GUIDE.md** with detailed exercises and objectives.

### 📖 Module Learning Guides

Each folder contains a comprehensive learning guide. Follow them sequentially or jump to areas of interest:

| Module | Focus Area | Learning Guide | Time Estimate |
|--------|------------|----------------|---------------|
| **00** | Environment Setup | [LEARNING_GUIDE](00_env_setup/LEARNING_GUIDE.md) | 1-2 hours |
| **01** | Python Programming | [LEARNING_GUIDE](01_python_programming/LEARNING_GUIDE.md) | 4-6 hours |
| **02** | Deep Learning Basics | [LEARNING_GUIDE](02_deep_learning_basics/LEARNING_GUIDE.md) ⭐ | 10-15 hours |
| **03** | TTS Systems | [LEARNING_GUIDE](03_tts_systems/LEARNING_GUIDE.md) | 6-8 hours |
| **04** | Speech/Audio Processing | [LEARNING_GUIDE](04_speech_audio_processing/LEARNING_GUIDE.md) | 5-7 hours |
| **05** | NLP | [LEARNING_GUIDE](05_nlp/LEARNING_GUIDE.md) | 6-8 hours |
| **06** | Hugging Face Transformers | [LEARNING_GUIDE](06_hf_transformers/LEARNING_GUIDE.md) | 8-12 hours |
| **07** | Data Preparation | [LEARNING_GUIDE](07_data_preparation/LEARNING_GUIDE.md) | 10-15 hours |
| **08** | Model Training & Fine-tuning | [LEARNING_GUIDE](08_model_training_finetuning/LEARNING_GUIDE.md) | 15-25 hours |
| **09** | Generative AI | [LEARNING_GUIDE](09_generative_ai/LEARNING_GUIDE.md) | 6-10 hours |
| **10** | MLOps | [LEARNING_GUIDE](10_mlops/LEARNING_GUIDE.md) | 8-12 hours |
| **11** | Cloud Platforms | [LEARNING_GUIDE](11_cloud_platforms/LEARNING_GUIDE.md) | 10-15 hours |
| **12** | Project Building | [LEARNING_GUIDE](12_project_building/LEARNING_GUIDE.md) | 15-20 hours |
| **13** | **Capstone Project** | [LEARNING_GUIDE](capstone_voice_replication_pipeline/LEARNING_GUIDE.md) 🎯 | 7 weeks |

⭐ = Foundational module - start here if new to ML  
🎯 = Final integrative project

**Total Estimated Time**: 100-150 hours + capstone (7 weeks)

### Repository Structure
```
voice-cloning-pipeline/
├── 00_env_setup/              # Initial environment setup
├── 01_python_programming/     # Python basics & utilities
├── 02_deep_learning_basics/   # PyTorch, neural networks, training loops ⭐
│   ├── README.md              # Module overview
│   ├── LEARNING_GUIDE.md      # Educational approach
│   ├── ML_CONCEPTS_EXPLAINED.md  # Deep dive into concepts (30KB)
│   ├── dl_utils.py            # Reusable utilities
│   └── 01-05 Python scripts   # Hands-on exercises
├── 03_tts_systems/            # Text-to-speech implementation
├── 04_speech_audio_processing/
├── 05_nlp/
├── 06_hf_transformers/
├── 07_data_preparation/
│   └── data/                  # Datasets (not in git - see setup below)
├── 08_model_training_finetuning/
├── 09_generative_ai/
├── 10_mlops/
├── 11_cloud_platforms/
├── 12_project_building/
└── capstone_voice_replication_pipeline/
**Total Estimated Time**: 100-150 hours + capstone (7 weeks)

---

## 🎯 How to Use This Repository

### For Complete Beginners
1. **[00_env_setup](00_env_setup/LEARNING_GUIDE.md)** - Set up your development environment
2. **[01_python_programming](01_python_programming/LEARNING_GUIDE.md)** - Learn Python for ML
3. **[02_deep_learning_basics](02_deep_learning_basics/LEARNING_GUIDE.md)** - Master core ML concepts ⭐
   - Also read: [ML_CONCEPTS_EXPLAINED.md](02_deep_learning_basics/ML_CONCEPTS_EXPLAINED.md) (comprehensive reference)
4. **Continue sequentially** through modules 03-12
5. **[Capstone](capstone_voice_replication_pipeline/LEARNING_GUIDE.md)** - Build your voice cloning system

### For Intermediate Learners
- **Have Python experience?** Start at [02_deep_learning_basics](02_deep_learning_basics/LEARNING_GUIDE.md)
- **Know PyTorch?** Jump to [03_tts_systems](03_tts_systems/LEARNING_GUIDE.md)
- **Want to deploy?** Focus on [10_mlops](10_mlops/LEARNING_GUIDE.md) and [11_cloud_platforms](11_cloud_platforms/LEARNING_GUIDE.md)

### For Advanced Users
- Review specific modules for gaps in knowledge
- Jump to [Capstone](capstone_voice_replication_pipeline/LEARNING_GUIDE.md) for the integrated project
- Use [ML_CONCEPTS_EXPLAINED.md](02_deep_learning_basics/ML_CONCEPTS_EXPLAINED.md) as a reference

---

## 🗺️ Learning Path Visualization

```
+---------------------------------------------------------------+
|                  VOICE CLONING PIPELINE                       |
|                  (Your Ultimate Goal)                         |
+---------------------------------------------------------------+
                            ^
                            |
          +-----------------+-----------------+
          |                                   |
    +-------------+                   +--------------+
    | TTS SYSTEMS |                   |  NLP & ML    |
    | (Mod 3-9)   |                   |  (Mod 5-9)   |
    +-------------+                   +--------------+
          |                                   |
          +-----------------+-----------------+
                            |
                    +---------------+
                    | DEEP LEARNING |
                    |    BASICS     |
                    | (Module 02)   | <-- START HERE
                    +---------------+
                            |
                    +---------------+
                    | PYTHON & ENV  |
                    | (Modules 0-1) |
                    +---------------+
```

---

### Learning Resources in Each Module

Each module folder contains:

1. **README.md** - Quick overview of the module
2. **LEARNING_GUIDE.md** - Detailed exercises, tasks, and learning objectives ⭐
3. **Code/Scripts** - Hands-on practice materials
4. **Data folders** - Sample datasets (when applicable)

**Special Resources:**
- [02_deep_learning_basics/ML_CONCEPTS_EXPLAINED.md](02_deep_learning_basics/ML_CONCEPTS_EXPLAINED.md) - 30KB comprehensive ML concept reference

---

## Your Learning Journey

```
Learning Path Progress:
    |
    v
[x] 00_env_setup
[x] 01_python_programming  
[>] 02_deep_learning_basics <-- Foundational module (START HERE)
[ ] 03_tts_systems
[ ] 04_speech_audio_processing
[ ] 05_nlp
[ ] 06_hf_transformers
[ ] 07_data_preparation
[ ] 08_model_training_finetuning
[ ] 09_generative_ai
[ ] 10_mlops
[ ] 11_cloud_platforms
[ ] 12_project_building
[ ] capstone_voice_replication_pipeline
```

**Current Focus**: Deep Learning Basics - Build the foundation for all subsequent modules

**Next Destination**: Text-to-Speech Systems - Apply DL to voice synthesis

---

## 📌 Summary

### Three Pillars of This Learning Path:

1. **Theory** - Comprehensive guides in each module's LEARNING_GUIDE.md
2. **Practice** - Hands-on Python scripts and exercises
3. **Projects** - Real-world applications culminating in the capstone

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional but recommended)
- Basic command-line knowledge

### Quick Start
1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/voice-cloning-pipeline.git
   cd voice-cloning-pipeline
   ```

2. **Start with environment setup**
   ```bash
   cd 00_env_setup
   # Follow instructions in README.md
   ```

3. **Begin learning path**
   ```bash
   cd ../02_deep_learning_basics
   python 01_hello_pytorch.py
   ```

### Data Setup

**Note:** The `data/` folders are excluded from git via `.gitignore` to keep the repository lightweight. You must manually download or generate required datasets.

#### Downloading LibriTTS Sample Data (for TTS training)

1. Visit [LibriTTS on OpenSLR](https://www.openslr.org/60/) or [LibriTTS on Hugging Face](https://huggingface.co/datasets/lj1995/LibriTTS)
2. Download desired subset (e.g., `dev-clean`)
3. Extract to `07_data_preparation/data/libritts_sample/LibriTTS/`
4. Your structure should look like:
   ```
   07_data_preparation/
   └── data/
       └── libritts_sample/
           └── LibriTTS/
               └── dev-clean/
   ```

**Important:** Do not add large datasets or audio files to git!

---

## 📖 Capstone Project: Custom Voice Replication Pipeline

**Goal:** Build an end-to-end pipeline that processes text and generates speech in your own voice.

**Key Steps:**
1. **Data Collection** - Record voice samples (10 short + 10 long)
2. **Preprocessing** - Trim, normalize, convert to WAV 24kHz mono
3. **Feature Extraction** - Extract MFCCs and spectrograms with librosa
4. **Model Training** - Train TTS model (Tacotron/FastSpeech/Coqui TTS)
5. **Text Processing** - Use NLP to clean and segment input
6. **Inference** - Generate speech from new text
7. **Deployment** - Build REST API (Flask/FastAPI)
8. **Evaluation** - Test quality and iterate

**Outcome:** A modular, working system that can read any text in your own voice, with all core AI skills practiced through manageable tasks.

---

## 🎯 AI Spheres Covered

This project touches multiple areas of AI:

| AI Sphere | Relevance to Voice Cloning | Technologies |
|-----------|---------------------------|--------------|
| **Deep Learning** ⭐ | Neural networks for speech synthesis | PyTorch, TensorFlow |
| **Speech & Audio Processing** ⭐ | TTS, voice conversion, audio enhancement | Coqui TTS, Bark, librosa |
| **NLP** ⭐ | Text parsing, summarization, dialogue detection | Hugging Face, spaCy |
| **Generative AI** ⭐ | Synthesizing unique voices, creating content | GPT, Bark, Stable Diffusion |
| **MLOps** | Model deployment, monitoring, automation | MLflow, Docker, FastAPI |
| **Cloud Platforms** | Scalable deployment | AWS, GCP, Azure |
| **Supervised Learning** | Audio classification, quality control | Scikit-learn |
| **Multi-modal AI** | Syncing narration with visuals | CLIP, Transformers |

⭐ = Core focus areas for voice cloning

---

## 📝 Learning Approach

This repository uses a **task-based learning approach** with:
- ✅ Hands-on Python scripts with extensive educational comments
- ✅ Comprehensive concept explanations ([ML_CONCEPTS_EXPLAINED.md](02_deep_learning_basics/ML_CONCEPTS_EXPLAINED.md))
- ✅ Clear learning objectives and expected outcomes
- ✅ Incremental complexity (basic → advanced)
- ✅ Real-world project structure

---

Now you know exactly where you are and where you're going. Happy learning! 🚀
