# RAG Document Assistant Testing Guide

This project includes real public documents in `sample_documents/` so you can test the assistant with authentic source material.

## Documents

1. `sample_documents/ai_governance_hourglass_model.pdf`
   - Source: arXiv paper, "Putting AI Ethics into Practice: The Hourglass Model of Organizational AI Governance"
   - URL: https://arxiv.org/abs/2206.00335

2. `sample_documents/ml_researcher_ai_ethics_survey.pdf`
   - Source: arXiv paper, "Ethics and Governance of Artificial Intelligence: Evidence from a Survey of Machine Learning Researchers"
   - URL: https://arxiv.org/abs/2105.02117

3. `sample_documents/nvidia_2024_10k_sec_filing.txt`
   - Source: NVIDIA fiscal 2024 Form 10-K from the U.S. SEC EDGAR archive
   - URL: https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/0001045810-24-000029.txt
   - Note: this file is large and HTML/XBRL-heavy. Use it as a stress test after the PDFs work.

## How To Test In The UI

1. Open the Streamlit app:
   - http://127.0.0.1:8501

2. Upload one or more files from `sample_documents/`.

3. Click `Build / Rebuild Index`.

4. Ask the questions below.

5. Check three things in every answer:
   - The answer should be grounded in the uploaded document.
   - Citations should point to the source file and page/chunk.
   - If the answer is not in the documents, it should say that it was not found instead of hallucinating.

## Quick API Test

Index the two clean PDF documents:

```powershell
curl.exe -s -X POST http://127.0.0.1:8010/index `
  -F "files=@sample_documents/ai_governance_hourglass_model.pdf" `
  -F "files=@sample_documents/ml_researcher_ai_ethics_survey.pdf"
```

Ask a question:

```powershell
curl.exe -s -X POST http://127.0.0.1:8010/ask `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"What are the three layers in the hourglass model of organizational AI governance?\"}"
```

## Questions For `ai_governance_hourglass_model.pdf`

1. What are the three layers in the hourglass model of organizational AI governance?
   - Expected evidence: environmental layer, organizational layer, and AI system layer.

2. What does the hourglass metaphor represent in AI governance?
   - Expected evidence: governance requirements flow from the environmental layer through the organization into AI systems.

3. Which two themes make up the organizational layer of AI governance?
   - Expected evidence: strategic alignment and value alignment.

4. Why do the authors say organizations need AI governance across the entire AI system life cycle?
   - Expected evidence: organizations must govern AI systems over development, deployment, operation, monitoring, and retirement, not only design.

5. What role does the paper assign to a Head of AI?
   - Expected evidence: overseeing AI system development and operations with authority, resources, and knowledge.

## Questions For `ml_researcher_ai_ethics_survey.pdf`

1. How many AI/ML researchers were surveyed?
   - Expected evidence: N = 524.

2. Which institutions did surveyed researchers trust more for shaping AI in the public interest?
   - Expected evidence: international organizations and scientific organizations.

3. How did respondents feel about AI/ML researchers working on lethal autonomous weapons?
   - Expected evidence: respondents were overwhelmingly opposed.

4. What did the survey find about AI safety research prioritization?
   - Expected evidence: a strong majority supported prioritizing AI safety research.

5. What pre-publication practice did many respondents support?
   - Expected evidence: ML institutions should conduct pre-publication review to assess potential harms.

## Questions For `nvidia_2024_10k_sec_filing.txt`

Use these after the two PDFs work, because the SEC file is much larger.

1. What were NVIDIA's reportable segments in fiscal 2024?
   - Expected evidence: Compute & Networking and Graphics.

2. What major demand trend does NVIDIA describe for Data Center revenue?
   - Expected evidence: strong demand related to data center computing platforms and generative AI / accelerated computing.

3. What supply chain risks does NVIDIA discuss?
   - Expected evidence: dependence on suppliers, foundries, assembly/test partners, capacity constraints, and component availability.

4. What fiscal year ended date is used in the filing?
   - Expected evidence: January 28, 2024.

5. What should the assistant do if you ask for something not in the filing, such as "What is NVIDIA's stock price today?"
   - Expected behavior: say it is not found in the document, because live stock price is not in the filing.

## Negative Tests

Ask these after indexing only the two arXiv PDFs:

1. What is NVIDIA's fiscal 2024 revenue?
   - Expected behavior: not found, unless the NVIDIA SEC file is also indexed.

2. What is the capital of France?
   - Expected behavior: not found in the uploaded documents.

3. Who won the latest IPL match?
   - Expected behavior: not found in the uploaded documents.
