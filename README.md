# 🚀 AI-Powered Resume Segregator Pro

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0%20Pro-orange.svg)]()

> **Professional AI-powered platform for intelligent candidate ranking and resume analysis using Google Gemini AI**

## ✨ Features

### 🎯 **Intelligent Job Analysis**
- **AI-Powered JD Parsing**: Automatically extracts job requirements, skills, and constraints
- **Smart Skill Extraction**: Identifies must-have vs. nice-to-have qualifications
- **Experience Requirements**: Automatically detects and validates experience levels
- **Education Matching**: Intelligent education requirement analysis

### 🚀 **Advanced Resume Scoring**
- **Semantic Similarity**: Uses Google's text embeddings for deep content understanding
- **Skill Coverage Analysis**: Comprehensive skill matching and gap analysis
- **Experience Validation**: Years of experience detection and validation
- **Composite Scoring**: Multi-factor weighted scoring algorithm

### 📊 **Professional Dashboard**
- **Real-time Analytics**: Live metrics and performance indicators
- **Interactive Visualizations**: Score distributions and skill coverage charts
- **Advanced Filtering**: Score-based and criteria-based candidate filtering
- **Export Capabilities**: CSV export with detailed analysis results

### 🔒 **Enterprise-Grade Security**
- **Local Processing**: All files processed locally, no data sent to external servers
- **Google AI Integration**: Secure API-based integration with Google Gemini
- **Privacy-First**: Your data stays on your infrastructure

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Modern web interface)
- **AI Engine**: Google Gemini 1.5 Flash
- **Text Processing**: PyPDF, advanced NLP techniques
- **Data Analysis**: Pandas, NumPy
- **Embeddings**: Google Text Embedding API
- **Language**: Python 3.8+

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Cloud API key with Gemini access
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/resume-segregator-pro.git
   cd resume-segregator-pro
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Google API key**
   ```bash
   export GOOGLE_API_KEY="your_actual_api_key_here"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Upload Job Description
- Upload a single PDF containing the job description
- The AI will automatically analyze and extract requirements

### Step 2: Upload Resumes
- Upload a ZIP file containing multiple PDF resumes
- Supports batch processing of unlimited resumes

### Step 3: AI Analysis
- Automatic constraint extraction from JD
- Semantic similarity scoring
- Skill coverage analysis
- Experience validation

### Step 4: Results & Export
- Ranked candidate list
- Detailed scoring breakdown
- Interactive analytics dashboard
- CSV export functionality

## 🎨 Features in Detail

### **Smart Scoring Algorithm**
```
Composite Score = (Similarity Weight × Semantic Similarity) + 
                  (Skills Weight × (0.6 × Skill Score + 0.3 × Experience + 0.1 × Education))
```

### **Advanced Analytics**
- **Score Distribution**: Visual breakdown of candidate performance
- **Skill Coverage**: Must-have vs. nice-to-have skill analysis
- **Top Performers**: Highlight best-matching candidates
- **Performance Metrics**: Average scores, top scores, candidate counts

### **Professional UI/UX**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern Styling**: Professional color scheme and typography
- **Interactive Elements**: Hover effects, progress bars, real-time updates
- **Accessibility**: Clear navigation and user feedback

## 🔧 Configuration

### Performance Settings
- **Max Pages per PDF**: Control processing time vs. accuracy
- **Max Characters**: Optimize for large documents
- **Scoring Weights**: Customize similarity vs. skills importance

### AI Model Settings
- **Embedding Model**: Google's text-embedding-004
- **Generation Model**: Gemini 1.5 Flash for fast analysis

## 📊 Sample Output

The application generates comprehensive reports including:

| Metric | Description |
|--------|-------------|
| **Composite Score** | Overall candidate fit (0.0 - 1.0) |
| **Semantic Similarity** | Content relevance score |
| **Skill Coverage** | Must-have skills match percentage |
| **Experience Match** | Years of experience validation |
| **Education Score** | Educational requirement alignment |
| **Overall Fit** | Qualitative assessment (Poor/Fair/Good/Excellent) |

## 🚀 Advanced Features

### **Batch Processing**
- Process hundreds of resumes simultaneously
- Intelligent error handling for corrupted files
- Progress tracking and status updates

### **Custom Scoring**
- Adjustable weights for different criteria
- Industry-specific scoring algorithms
- Custom skill dictionaries

### **Export & Reporting**
- CSV export with all metrics
- Timestamped file naming
- Professional report formatting

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/resume-segregator-pro.git
cd resume-segregator-pro
pip install -r requirements-dev.txt
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for powerful language understanding
- **Streamlit** for the excellent web framework
- **Open Source Community** for various Python packages

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/resume-segregator-pro/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/resume-segregator-pro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/resume-segregator-pro/discussions)

## 🔮 Roadmap

- [ ] **Multi-language Support**: Support for non-English resumes
- [ ] **Advanced Analytics**: Machine learning-based candidate insights
- [ ] **Integration APIs**: Connect with ATS and HR systems
- [ ] **Mobile App**: Native iOS and Android applications
- [ ] **Enterprise Features**: Multi-user, role-based access control

---

<div align="center">
  <p><strong>Built with ❤️ for HR professionals and recruiters</strong></p>
  <p>Transform your hiring process with AI-powered insights</p>
</div>

