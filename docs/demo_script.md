# Demo Video Script

**Duration:** 3 minutes

**0:00 — What is MLForge?**
"Welcome to MLForge. MLForge is an autonomous, multi-agent AI system designed to intelligently orchestrate machine learning pipelines end-to-end."

**0:20 — Architecture**
"Instead of relying on a single, monolithic AI, MLForge assigns highly specialized agents—like Data Analysts, Feature Engineers, and Model Trainers—to strictly defined roles, communicating securely through a shared state."

**0:45 — Upload dataset**
*(Show UI Dataset upload)*
"Let's upload a small dataset, for example, `heart_disease_500.csv`. MLForge analyzes the columns and metadata instantly."

**1:00 — Define objective**
"We want to predict whether a patient has heart disease. We select 'Classification' and prioritize 'Recall'."

**1:15 — Start workflow**
*(Click RUN MLFORGE)*
"Once started, the Orchestrator plans the experiment."

**1:30 — Agent trace**
*(Scroll down to the Agent Trace dashboard)*
"Here we see the live observability trace. The Data Analyst agent profiled the data, and the Feature Engineer recommended imputation for missing values. Now, the Training agent is writing and executing Python code via the sandboxed Tool system."

**2:00 — Model comparison**
*(Show final models)*
"The experiments finish. MLForge trained Logistic Regression and Random Forest models."

**2:20 — Evaluation**
"The Evaluation Agent benchmarks both models against our required metric: Recall."

**2:40 — Reviewer**
*(Show Reviewer tab)*
"Finally, the Reviewer Agent independently audits the pipeline. It checks for data leakage and overfitting, and approves the model."

**3:00 — Final result**
"We have a validated, trained ML model ready for deployment."

**3:20 — Reliability/observability**
"And if a provider like Gemini had failed, MLForge's fallback system would have transparently routed the prompt to Groq, ensuring the pipeline never halts."
