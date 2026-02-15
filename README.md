# Voice Cloning Pipeline 🎙️

A hands-on learning path for building a modern voice cloning pipeline using deep learning, TTS systems, and related open-source tools. This project is designed for learning, experimentation, and developing AI engineering skills through practical implementation.

---

## �️ Interactive Learning Visualization

**[Explore the AI Learning Path Graph](ai_mind_map/ai_complete_learning_path.html)** - Interactive D3.js visualization showing:
- 10 major AI learning domains
- 60 specific topics and techniques
- Your current learning progress (✅ completed / 📋 planned)
- Foundational relationships and prerequisites
- Direct links to relevant folders and resources

**Tip:** Open `ai_mind_map/ai_complete_learning_path.html` in your browser to see your learning journey!

---

## 📚 Repository Structure

This repository is organized by **AI learning domains** rather than sequential modules, allowing you to navigate based on concepts and build knowledge non-linearly.

### 🎯 Learning Domains

| Domain | Status | Description |
|--------|--------|-------------|
| **[01_mathematics](01_mathematics/)** | 📋 Planned | Linear algebra, calculus, probability, information theory |
| **[02_machine_learning](02_machine_learning/)** | ✅ Active | Supervised/unsupervised learning, feature engineering, evaluation |
| **[03_deep_learning](03_deep_learning/)** | ✅ Active | Neural networks, CNNs, RNNs, optimization, regularization |
| **[04_computer_vision](04_computer_vision/)** | 📋 Planned | Image classification, object detection, segmentation |
| **[05_nlp](05_nlp/)** | ✅ Active | Text processing, embeddings, language models, **speech & audio** |
| **[06_reinforcement_learning](06_reinforcement_learning/)** | 📋 Planned | MDPs, Q-learning, policy gradients, RLHF |
| **[07_tools_frameworks](07_tools_frameworks/)** | ✅ Active | Python, PyTorch, TensorFlow, Hugging Face, dev tools |
| **[08_data_engineering](08_data_engineering/)** | ✅ Active | Data collection, preprocessing, pipelines, storage |
| **[09_research_advanced](09_research_advanced/)** | 📋 Planned | Meta-learning, NAS, multimodal AI, interpretability |
| **[10_ai_ethics](10_ai_ethics/)** | 📋 Planned | Fairness, privacy, alignment, responsible AI |

**Supporting Resources:**
- **[ai_mind_map/](ai_mind_map/)** - Interactive visualizations and learning guides
- **[capstone_projects/](capstone_projects/)** - Integrative voice cloning project

---

## 🚀 Quick Start Paths

### Path 1: Complete Beginner (Python → ML → DL → TTS)
1. [Python Basics](07_tools_frameworks/python_basics/)
2. [Machine Learning Fundamentals](02_machine_learning/)
3. [Deep Learning](03_deep_learning/)
4. [NLP & Speech/Audio](05_nlp/speech_audio/)
5. [Voice Cloning Project](capstone_projects/voice_replication_pipeline/)

### Path 2: ML Practitioner → Voice AI Specialist  
1. [Deep Learning Neural Networks](03_deep_learning/neural_networks/)
2. [Deep Learning Optimization](03_deep_learning/optimization/)
3. [NLP Speech & Audio](05_nlp/speech_audio/tts/)
4. [Data Engineering for Audio](08_data_engineering/)
5. [Voice Cloning Project](capstone_projects/voice_replication_pipeline/)

### Path 3: Explore by Interest
Use the [interactive learning graph](ai_mind_map/ai_complete_learning_path.html) to:
- Click on topics you're interested in
- See prerequisites and related areas
- Navigate directly to relevant code and resources

---

## ✅ Current Learning Progress

### Completed Domains
- **Know PyTorch?** Jump to [03_tts_systems](03_tts_systems/LEARNING_GUIDE.md)
- **Want to deploy?** Focus on [10_mlops](10_mlops/LEARNING_GUIDE.md) and [11_cloud_platforms](11_cloud_platforms/LEARNING_GUIDE.md)

### For Advanced Users
- Review specific modules for gaps in knowledge
- Jump to [Capstone](capstone_voice_replication_pipeline/LEARNING_GUIDE.md) for the integrated project
- Use [ML_CONCEPTS_EXPLAINED.md](02_deep_learning_basics/ML_CONCEPTS_EXPLAINED.md) as a reference

---

## 🗺️ Learning Path Visualization


```text
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
### Completed Domains

**Machine Learning** ✅
- Logistic regression and classification fundamentals
- Supervised learning lab completed

**Deep Learning** ✅
- 9 comprehensive labs covering:
  - Neural networks (feedforward, backpropagation, MNIST)
  - Hidden layer networks, L-layered architectures
  - Optimization techniques (Adam, gradient descent, initialization)
  - Regularization methods (dropout, L2, batch normalization)
  - Gradient checking and debugging
  - PyTorch and TensorFlow implementations

**NLP / Speech & Audio** ✅
- Text-to-speech systems with voice cloning
- Bark TTS experiments
- XTTS custom voice demos
- Batch audio processing utilities

**Tools & Frameworks** ✅
- Python programming (file I/O, data processing)
- PyTorch fundamentals
- TensorFlow basics
- Jupyter notebooks for experimentation
- Git and development workflows

**Data Engineering** ✅
- Data collection and preprocessing scripts
- Audio format conversion tools
- Dataset preparation pipelines

### 📋 Planned Topics
- Mathematics foundations (linear algebra, calculus, statistics)
- Computer vision (image classification, object detection)
- Reinforcement learning (Q-learning, policy gradients)
- Advanced transformers and language models
- Hugging Face ecosystem
- MLOps and deployment
- Cloud platforms (AWS, GCP, Azure)
- AI ethics and responsible AI

---

## 📂 Detailed Repository Structure

```
voice-cloning-pipeline/
│
├── 00_foundations/
│   └── environment_setup/          # Dev environment configuration
│
├── 01_mathematics/                 # 📋 Math fundamentals (planned)
│   ├── linear_algebra/
│   ├── calculus/
│   ├── probability_statistics/
│   └── information_theory/
│
├── 02_machine_learning/            # ✅ ML algorithms & techniques
│   ├── supervised_learning/        # Logistic regression lab
│   ├── feature_engineering/
│   └── model_evaluation/
│
├── 03_deep_learning/               # ✅ Neural networks & DL
│   ├── neural_networks/            # 5 Python scripts, 3 labs
│   ├── optimization/               # 4 labs (init, gradient checking, optimization)
│   ├── regularization/             # Regularization techniques lab
│   ├── cnns/                       # Planned
│   ├── rnns/                       # Planned
│   └── transformers/               # Planned
│
├── 04_computer_vision/             # 📋 Image processing (planned)
│
├── 05_nlp/                         # ✅ Natural language processing
│   └── speech_audio/
│       └── tts/                    # 6 TTS scripts and demos
│
├── 06_reinforcement_learning/      # 📋 RL algorithms (planned)
│
├── 07_tools_frameworks/            # ✅ Development tools
│   ├── python_basics/              # 3 scripts, 4 notebooks
│   ├── pytorch/                    # dl_utils.py
│   ├── tensorflow/                 # TensorFlow intro lab
│   ├── hugging_face/               # Planned
│   └── dev_tools/
│
├── 08_data_engineering/            # ✅ Data processing
│   └── data_preprocessing/
│       └── scripts/                # Audio data preparation
│
├── 09_research_advanced/           # 📋 Advanced topics (planned)
│
├── 10_ai_ethics/                   # 📋 Responsible AI (planned)
│
├── ai_mind_map/                    # 🗺️ Interactive visualizations
│   ├── ai_complete_learning_path.html     # Main interactive graph
│   ├── ai_learning_mindmap_graph.html     # Voice cloning specific
│   ├── GRAPH_DESIGN_GUIDELINES.md         # Design decisions
│   └── GITHUB_PAGES_SETUP.md              # Deployment guide
│
├── capstone_projects/
│   └── voice_replication_pipeline/        # Integrative project
│
└── _archive_old_structure/        # Old sequential module organization
```

---

## 🎓 How to Use This Repository

### 1. **Visual Navigation (Recommended)**
Open [ai_mind_map/ai_complete_learning_path.html](ai_mind_map/ai_complete_learning_path.html) in your browser:
- See your learning progress at a glance
- Click on topics to view details and resources
- Understand prerequisite relationships
- Navigate directly to relevant folders

### 2. **Domain-Based Learning**
Browse by topic of interest:
- Want to learn neural networks? → `03_deep_learning/neural_networks/`
- Interested in speech synthesis? → `05_nlp/speech_audio/tts/`
- Need Python practice? → `07_tools_frameworks/python_basics/`

### 3. **Sequential Path (Traditional)**
Follow the learning progression:
1. Mathematics foundations (if needed)
2. Machine Learning basics
3. Deep Learning core concepts
4. Specialized applications (CV, NLP, RL)
5. Tools & frameworks as needed
6. Capstone project integration

### Learning Resources in Each Folder
- **README.md** - Overview, status, key topics, prerequisites
- **Scripts & notebooks** - Hands-on practice materials
- **Labs** - Structured exercises with objectives
- Links to related topics and external resources

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 2.0+ / TensorFlow (depending on your path)
- CUDA-capable GPU (optional but recommended for DL)
- Git and basic command-line knowledge

### Quick Start

**Option 1: Explore Visually**
```bash
git clone https://github.com/Plamen5kov/voice-cloning-pipeline.git
cd voice-cloning-pipeline
# Open ai_mind_map/ai_complete_learning_path.html in your browser
```

**Option 2: Start with Python Basics**
```bash
cd 07_tools_frameworks/python_basics
python convert_m4a_to_wav.py  # Example script
jupyter notebook  # Explore notebooks
```

**Option 3: Jump to Deep Learning**
```bash
cd 03_deep_learning/neural_networks
python 01_hello_pytorch.py  # PyTorch basics
python 03_train_mnist.py    # Train your first neural network
```

**Option 4: TTS and Voice Cloning**
```bash
cd 05_nlp/speech_audio/tts
python tts_basic_demo.py    # Text-to-speech demo
```

---

## 📊 Repository Statistics

- **10 Learning Domains** covering the full AI landscape
- **60+ Specific Topics** in the learning graph
- **15+ Labs** completed (neural networks, optimization, regularization)
- **10+ Python Scripts** for hands-on practice
- **4 Jupyter Notebooks** for experimentation
- **6 TTS Demos** for voice cloning
- **~2.6GB Data** (gitignored) for training and experimentation

---

## 📝 Notes on Repository Organization

### Recent Restructuring (February 2026)
This repository was recently reorganized from a **sequential module structure** (00-12) to a **domain-based structure** (01_mathematics through 10_ai_ethics) to:
- Better align with the [AI Learning Path visualization](ai_mind_map/ai_complete_learning_path.html)
- Enable non-linear, interest-driven learning
- Group related concepts together
- Show clear prerequisite relationships

**Old structure preserved in:** `_archive_old_structure/` (gitignored)

**Design decisions documented in:** `ai_mind_map/GRAPH_DESIGN_GUIDELINES.md`

**Migration plan available in:** `REPO_RESTRUCTURE_PLAN.md`

### Data Files
**Note:** `data/` folders are excluded from git to keep the repository lightweight.

Large datasets (LibriSpeech audio, preprocessed features, etc.) are stored locally but not committed. You may need to download or generate datasets for certain labs and projects.

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
