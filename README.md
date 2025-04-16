🧠 Fine-Tuning a Domain-Specific LLM that Outperformed GPT-4o & Gemini

This project explores how smaller, fine-tuned Language Models (LLMs) can outperform massive general-purpose models like GPT-4o and Gemini on domain-specific tasks. In this case, the task was to identify missing key points in answers generated from long-form content — a challenge where precision and contextual understanding are vital.

Instead of relying on brute-force parameter counts, this project leverages LoRA-based fine-tuning and domain-specific data to boost performance in a task where large models often return vague or incomplete responses.


---

🌟 Problem Statement

Given an answer generated from a long paragraph, the goal is to evaluate whether all required key points are present, and if not, identify the missing ones. This task is critical in domains like:

Education: Evaluating student answers.

Legal: Ensuring contract completeness.

Medical: Validating report summaries.


🔍 Example

Paragraph (excerpt):

> Air pollution has a wide range of health impacts including chronic bronchitis, heart disease, and eye irritation. Global studies have conclusively shown these effects even at sub-toxic levels...



Key Points:

Chronic bronchitis is a major health effect.

Air pollution impacts are proven by global studies.

Eye irritation occurs due to particulates.

Sub-toxic levels are still dangerous.


Answer:

> Air pollution causes chronic bronchitis and is harmful even at low levels. Global studies support this.



Expected Output:

{
  "Points_Missed": ["Eye irritation occurs due to particulates"]
  }


  ---

  📌 Project Highlights

  📔 Explore the dataset creation in Dataset/dataset_creation.ipynb

  🔧 Fine-tuning notebook is available at Training/finetune.ipynb

  🧪 Evaluation notebook can be found at Evaluation/evaluate.ipynb

  🚀 Fine-tuned using the LoRA technique on a domain-specific dataset.

  📊 Outperformed GPT-4o and Gemini on precision, recall, and F1-score.

  📁 Includes dataset creation, training, and evaluation pipeline.

  🔍 Focused on identifying missing key points from generated answers.



  ---

  🗂️ Project Structure

  .
  ├── Dataset/
     ├── your_dataset.csv              # Original dataset with paragraphs
     │   ├── output.json                   # Generated QA pairs with missing key points
     │   ├── formatted_train_data.json     # Alpaca-style training data
     ├── Training/
     │   ├── finetune.ipynb                # Fine-tuning using Unsloth & LoRA
     ├── Evaluation/
     │   ├── evaluate.ipynb                # Custom evaluation pipeline
     ├── Results/
     │   ├── *.png                         # Performance visualization images


     ---

     🧾 Dataset Creation

     The base dataset was sourced from the Stanford Question Answering Dataset (SQuAD).

     We used OpenAI GPT-4o via Azure to synthetically generate QA pairs from long paragraphs:

     Extracted 2–5 key points from a paragraph.

     Randomly removed 1–4 key points to simulate incomplete answers.

     Generated JSON in the format:


     {
       "Answer": "paragraph with missing key points",
         "key_points": ["point1", "point2"],
           "PointsMissed": ["missing_point1"]
           }

           These were converted to Alpaca-style prompts for training.


           ---

           ⚙️ Fine-Tuning Pipeline

           Used Unsloth for efficient LoRA fine-tuning.

           Trained on formatted Alpaca-style prompts with 300+ examples.

           Model used: Meta-Llama-3.1-8B

           Training config:

           Batch Size: 2

           Epochs: 1

           Optimizer: adamw_8bit

           Sequence Length: 2048

           LoRA rank: 16




           ---

           🥪 Evaluation

           The model was evaluated by comparing predicted PointsMissed with the ground truth.

           📐 Metrics Calculation

           Precision: Measures how many predicted points were actually correct.

           Precision = True Positives / (True Positives + False Positives)

           Recall: Measures how many of the true missing points were recovered.

           Recall = True Positives / (True Positives + False Negatives)

           F1-Score: Harmonic mean of precision and recall.

           F1 = 2 * (Precision * Recall) / (Precision + Recall)

            Evaluation Example

            Predicted: {"PointsMissed": ["Point A", "Point B"]}
            Ground Truth: {"PointsMissed": ["Point A", "Point C"]}

            True Positive: Point A

            False Positive: Point B

            False Negative: Point C



            ---

            📈 Results

            Our fine-tuned models were evaluated against GPT-4o, Gemini, and base models using precision, recall, and F1-score. Below is the performance comparison:

            > ✅ Best Performing Model: Mistral-7B Instruct Finetuned
            📉 Notably Outperformed: ChatGPT-4o, Gemini 2.0, and all base models
            🌟 Domain-specific fine-tuning showed consistent improvements in task performance.



            📊 Visualization of Results

             Comparison of precision, recall and F1 scores across all evaluated models

              Detailed performance breakdown showing all metrics for each evaluated model

               Heatmap visualization highlighting relative strengths and weaknesses across all models

                Bubble chart visualization showing precision vs recall with F1-score represented by bubble size


                ---

                💬 Key Takeaways

                Domain-specific fine-tuning can outperform even the most powerful foundation models on specific tasks.

                LoRA + Unsloth enables fast, memory-efficient training even on limited GPU resources like Kaggle T4.

                A clean evaluation pipeline is crucial for comparing models fairly.



                ---

                📦 Installation & Usage

                # Install requirements
                pip install unsloth

                # Clone this repo and follow notebooks step by step


                ---

                🧑💻 Author

                Vivek B S
                Final Year CSE (AI & ML) | Passionate about LLMs & domain-specific AI
                LinkedIn | GitHub


                ---

                