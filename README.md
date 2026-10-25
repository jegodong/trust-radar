# trust-radar-backup

### 1. Prerequisites (Install these first)
- Install [Python](https://www.python.org/downloads/) (Make sure to check "Add Python to PATH").
- Install [Ollama](https://ollama.com/) (Download the Windows/Mac installer and run it).

### 2. Download the Model
Open your terminal (PowerShell/CMD) and pull the default model:
- ollama pull qwen2.5:3b

### 3. Install Python Dependencies
Clone this repository or download the ZIP. Open your terminal in the project folder, and run:
- pip install -r requirements.txt

###4. Download the spaCy Language Model
This step is mandatory (requirements.txt cannot install it directly):
- python -m spacy download en_core_web_sm

###5. Run the Dashboard
- streamlit run dashboard.py
