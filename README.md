# Machine Learning Curriculum

This curriculum is designed to be a comprehensive roadmap for learning Machine Learning, from foundational knowledge to advanced specialization and deployment. It is structured for progressive learning.

---

### **Phase 0: The Bedrock - Foundational Knowledge**

These are the non-negotiable prerequisites.

*   **1. Mathematics:**
    *   **Linear Algebra:** The language of data.
        *   **Concepts:** Scalars, Vectors, Matrices, Tensors.
        *   **Operations:** Dot Products, Matrix Multiplication, Transpose, Inverse.
        *   **Theory:** Vector Spaces, Eigenvectors & Eigenvalues, Singular Value Decomposition (SVD).
        *   **Application:** Data representation, PCA, Neural Network operations.
    *   **Calculus:** The language of optimization.
        *   **Concepts:** Derivatives, Partial Derivatives, Gradients, The Chain Rule.
        *   **Theory:** Finding the minima/maxima of functions.
        *   **Application:** Understanding and implementing Gradient Descent.
    *   **Probability & Statistics:** The language of uncertainty and evaluation.
        *   **Probability:** Conditional Probability, Bayes' Theorem, Probability Distributions (Normal, Binomial).
        *   **Statistics:** Mean, Median, Variance, Standard Deviation, Correlation vs. Causation, Hypothesis Testing.
        *   **Application:** Understanding data, evaluating model performance.

*   **2. Programming & Tools:**
    *   **Python Programming:**
        *   **Fundamentals:** Data types, loops, conditionals, functions, classes (OOP).
        *   **Core Data Science Libraries:** NumPy, Pandas, Matplotlib, Seaborn.
    *   **Development Environment:**
        *   **Interactive:** Jupyter Notebooks / JupyterLab.
        *   **Editor:** VS Code.
    *   **Version Control:**
        *   **Tools:** Git & GitHub.

---

### **Phase 1: Core Machine Learning Fundamentals**

The heart of traditional machine learning.

*   **1. The Big Picture:**
    *   **Types of ML:** Supervised Learning, Unsupervised Learning, Reinforcement Learning.
    *   **Core Concepts:** Features, Labels, Training/Validation/Test Sets, Overfitting, Underfitting, The Bias-Variance Tradeoff.
    *   **The ML Workflow:** Data Collection -> Preprocessing -> Training -> Evaluation -> Deployment.

*   **2. Supervised Learning:**
    *   **Regression (Predicting continuous values):**
        *   Linear Regression
        *   Polynomial Regression
    *   **Classification (Predicting categories):**
        *   Logistic Regression
        *   k-Nearest Neighbors (k-NN)
        *   Support Vector Machines (SVMs)
        *   Naive Bayes
        *   Decision Trees
    *   **Ensemble Methods (Combining models):**
        *   Bagging & Random Forests
        *   Boosting: AdaBoost, Gradient Boosting Machines (GBM), XGBoost, LightGBM.

*   **3. Unsupervised Learning:**
    *   **Clustering (Grouping data):**
        *   K-Means Clustering
        *   Hierarchical Clustering
        *   DBSCAN
    *   **Dimensionality Reduction (Simplifying data):**
        *   Principal Component Analysis (PCA)
        *   t-SNE (for visualization).

*   **4. Model Development Lifecycle:**
    *   **Feature Engineering:** Handling Missing Data, Encoding Categorical Variables, Feature Scaling.
    *   **Model Evaluation (Metrics):**
        *   **Regression:** MAE, MSE, R-squared.
        *   **Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC/AUC.
    *   **Model Selection:** Cross-Validation, Hyperparameter Tuning (Grid Search, Random Search).

---

### **Phase 2: Specializations & Advanced Topics**

Diving deeper into modern and complex areas.

*   **1. Deep Learning (Neural Networks):**
    *   **Frameworks:** PyTorch or TensorFlow/Keras.
    *   **Foundations:** Perceptrons, Multi-Layer Perceptrons (MLPs), Backpropagation, Activation Functions (ReLU), Optimizers (Adam).
    *   **Computer Vision (CV):** Convolutional Neural Networks (CNNs). Key architectures: ResNet, EfficientNet.
    *   **Sequential Data:** Recurrent Neural Networks (RNNs), LSTMs, GRUs.
    *   **The Transformer Architecture:** The Attention Mechanism, BERT, GPT models.

*   **2. Natural Language Processing (NLP):**
    *   **Classic:** Bag-of-Words, TF-IDF.
    *   **Modern:** Word Embeddings (Word2Vec), Transformers.

*   **3. Reinforcement Learning (RL):**
    *   **Concepts:** Agent, Environment, State, Action, Reward.
    *   **Algorithms:** Q-Learning, Deep Q-Networks (DQN).

---

### **Phase 3: MLOps & Productionalization**

Taking models from a notebook to the real world.

*   **1. Deployment:** Creating APIs (Flask, FastAPI), Containerization (Docker).
*   **2. Automation:** Building ML pipelines (Kubeflow, Airflow).
*   **3. Cloud Platforms:** AWS SageMaker, Google Vertex AI, or Azure ML.
*   **4. Monitoring:** Model Drift, Data Drift, Data Version Control (DVC).

---

### **Phase 4: Continuous Learning & Application**

Staying current in a fast-evolving field.

*   **1. Practical Experience:** Kaggle competitions, personal end-to-end projects.
*   **2. Stay Updated:** Reading papers on arXiv, following major conferences (NeurIPS, ICML).
*   **3. Soft Skills:** Problem formulation, communication, and storytelling with data.
