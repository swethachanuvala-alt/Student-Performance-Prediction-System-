# Student Math Score Predictor (green theme)

## Run in VS Code (Windows cmd)
1. Open this folder in VS Code (File > Open Folder).
2. Open a terminal: Terminal > New Terminal, and choose "Command Prompt".
3. Run these commands:

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at http://localhost:8501. Stop it with Ctrl+C.
Shortcut: double-click `run.bat` (or type `run.bat` in cmd) to do all of the above.

## Deploy free (Streamlit Community Cloud)
Push `app.py`, `requirements.txt`, `StudentsPerformance.csv` and the `.streamlit` folder to GitHub,
then create the app at share.streamlit.io with main file `app.py`.

## Files
- app.py: web app (trains the model on startup, same steps as the notebook)
- StudentsPerformance.csv: dataset
- GGST_3.ipynb: your original notebook
- .streamlit/config.toml: green theme
