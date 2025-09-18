# DocuMentor: Automated QA Generation & Model Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

DocuMentor is an advanced pipeline for generating high-quality question-answer pairs from technical documents and fine-tuning language models for reasoning tasks. This project automates the process of creating training data from PDFs/Markdown files and prepares it for model training.

## Key Features

- **Automated QA Generation**: Extract meaningful Q&A pairs from technical documents
- **Quality Assessment**: AI-powered evaluation of question-answer quality
- **Fine-tuning Ready**: Prepare data for training reasoning models
- **Modular Design**: Easy to extend and customize components
- **Production-Ready**: Built with best practices for maintainability

## Project Structure

```
DocuMentor/
├── src/                    # Source code
│   ├── generate_qa.py      # Generate Q&A pairs from documents
│   ├── evaluate_qa.py      # Evaluate QA pair quality
│   └── prompt_templates.py # AI prompt templates
├── notebooks/              # Jupyter notebooks for model training
│   └── fine_tuning_model.ipynb
├── data/                   # Data directory (not versioned)
│   ├── raw/                # Raw input documents
│   └── processed/          # Processed datasets and results
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DocuMentor.git
   cd DocuMentor
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Prepare Your Document**:
   - Place your Markdown file in `data/raw/`
   - (Optional) Convert PDF to Markdown using your preferred tool

2. **Generate Q&A Pairs**:
   ```bash
   python src/generate_qa.py
   ```
   This will process your document and generate Q&A pairs in `data/processed/`

3. **Evaluate Quality**:
   ```bash
   python src/evaluate_qa.py
   ```
   This will evaluate and filter the generated Q&A pairs for quality

4. **Fine-tune Your Model**:
   Open `notebooks/fine_tuning_model.ipynb` to train your reasoning model with the generated data.

## Configuration

Customize the following in the respective files:
- `src/prompt_templates.py`: Modify the prompt templates for QA generation
- `src/generate_qa.py`: Adjust parameters like chunk size and quality thresholds
- `src/evaluate_qa.py`: Fine-tune the quality assessment criteria

