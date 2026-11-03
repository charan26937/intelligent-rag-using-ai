# Intelligent HR Policy RAG

A Streamlit app for asking questions about HR policy PDFs. It extracts PDF text,
splits it into overlapping passages, embeds the passages locally, and searches
them with FAISS. Answers include source-file and page citations.

## Run in PowerShell

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\data\hr_policies\faiss_index\requirements.txt
python -m streamlit run .\data\hr_policies\faiss_index\app.py
```

If PowerShell prevents activation, allow it for just the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The app loads PDF files from `data\hr_policies` and also accepts PDFs uploaded
in the sidebar. If a local `.pdf` file contains a path to the original document,
the app follows that path when it points inside the policy folder or your
`Downloads` folder. Scanned/image-only PDFs need OCR before they can be searched.
The first run downloads the sentence-transformer embedding model.

## Answer modes

- **Evidence-only:** Works without an API key and displays the closest matching
  passages with page citations.
- **Groq-generated:** Configure `GROQ_API_KEY` to generate a concise answer
  grounded in the retrieved passages. The default model is
  `openai/gpt-oss-120b`; override it with `GROQ_MODEL` if needed.

## LangSmith tracing

Set `LANGSMITH_API_KEY` to enable the optional tracing control in the sidebar.
When tracing is enabled, LangSmith receives the question, retrieved policy
passages, and generated answer. Leave tracing disabled if that data should not
be sent to LangSmith. The app reads these settings from the `faiss_index\.env`
file or the process environment:

```env
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=intelligent-hr-policy-rag
```

Do not commit `.env` or share API keys.
