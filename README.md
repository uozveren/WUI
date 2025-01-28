# WUI
This  application provides a user-friendly web interface for Unsloth to fine-tune Large Language Models (LLMs) with minimal friction. Whether you're exploring on CPU or harnessing the power of GPU, this app offers real-time logs, automated retry on out-of-memory (OOM) errors, advanced metrics and even a built-in TensorBoard panel.
✨ Key Features
CPU/GPU Selection
Decide where to run your training—on CPU or GPU. If you pick GPU, the app installs CUDA-enabled PyTorch dependencies automatically.
<br>🖥️ CPU or ⚡ GPU—your call!

Streamlined Installation
Install all required Python libraries with a single click from the interface.
<br>🛠️ One-click setup!

Dataset Management
Upload your training/testing .jsonl files directly.
<br>📂 No more manual file transfers!

Easy Config Creation
Generate or load config.yaml in just a few clicks. Adjust hyperparameters, resume checkpoints, or enable QLoRA flags.
<br>🔧 Just fill and go!

Real-Time Training Logs
Watch your Unsloth fine-tuning process via live streamed logs. Automatic retry on GPU out-of-memory by reducing batch size.
<br>📡 Stay updated, no terminal needed!

TensorBoard Integration
Start TensorBoard from the app, and view it inside Streamlit via an embedded iFrame or in a separate tab on port 6006.
<br>📊 Visualize training curves seamlessly!

Automatic Metrics & Visualizations
Evaluate your LLM with ROUGE, BLEU, plus generate Confusion Matrix and ROC curves for classification tasks.
<br>📈 🧩 All in the same interface.

Experiment Logging
Parameters and metric results for each run get stored in an experiment_logs.csv. Easily track multiple experiments, compare results, and plot them in-line with st.line_chart.
<br>📜 Build a history of all your runs.

🔧 Getting Started
Clone the Repository

bash
Kopyala
git clone https://github.com/<user>/unsloth-finetuning-webui.git
cd unsloth-finetuning-webui
Install & Launch

(Optional) Create and activate a Python virtual environment.
Install Streamlit (and optionally other dependencies):
bash
Kopyala
pip install streamlit
Run the app:
bash
Kopyala
streamlit run app.py
Access the web UI at http://localhost:8501.
Configure CPU/GPU

From the UI, select CPU or GPU. If GPU is chosen, the app can install CUDA-enabled packages automatically.
Upload Datasets

Use the file uploader to provide your training and test .jsonl datasets.
Create/Load config.yaml

Adjust model name, batch size, learning rate, and other hyperparameters from the interface.
Start Fine-Tuning

Click “Start Training.” Watch real-time logs. If a GPU OOM error occurs, the batch size auto-adjusts.
Evaluate & Visualize

Run evaluations with advanced metrics (ROUGE, BLEU).
Inspect Confusion Matrix and ROC plots directly in the app.
Track Experiments

Check or plot your results in experiment_logs.csv.
TensorBoard

Launch with one click, see the TensorBoard panel embedded in the Streamlit UI.
🤝 Contributing
Contributions are welcome! Feel free to:

Open an issue for any suggestions, bug reports, or questions.
Submit a pull request if you’d like to add new features or improve the codebase.
Please follow standard GitHub workflows and ensure your PR is well-documented and tested.

📄 License
MIT License. See LICENSE file for details.

Happy Fine-Tuning!
Build your custom LLM with ease—now with CPU/GPU flexibility, real-time metrics, TensorBoard, and more.






