# PII Detection and Redaction Tool Documentation

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Installing Dependencies](#installing-dependencies)
- [Running the PII Detection Script](#running-the-pii-detection-script)
  - [Script Overview](#script-overview)
  - [Script: `detect_pii.py`](#script-detect_piipy)
  - [Handling File Paths on Windows](#handling-file-paths-on-windows)
  - [Running the Script](#running-the-script)
- [Assessing Tool Performance](#assessing-tool-performance)
  - [Suggested Performance Metrics](#suggested-performance-metrics)
    - [Precision and Recall](#precision-and-recall)
    - [F1 Score](#f1-score)
  - [Evaluating the Tool](#evaluating-the-tool)
- [Including the Updated Dataset](#including-the-updated-dataset)
- [PII Detection on Dataset](#pii-detection-on-dataset)
- [Cloud Architecture](#cloud-architecture)
    - [Diagram](#diagram)
    - [Diagram Components](#diagram-components)
    - [Design Considerations and Decision-Making Process](#design-considerations-and-decision-making-process)
    - [Final Considerations](#final-considerations)

---

## Introduction

The **PII Detection and Redaction Tool** is designed to identify and redact Personally Identifiable Information (PII) from text data. This documentation provides detailed instructions on how to:

- Use the `PIIRedactor` class to detect PII in a dataset.
- Write and run a script to process your dataset and add new binary features indicating the presence of different PII types.
- Suggest appropriate performance metrics for assessing the tool's performance.
- Include the updated dataset in your repository.

---

## Project Structure

All necessary files are located in the `section3` folder of the repository:

```plaintext
Responsible-AI-Take-Home-Assessment/
├── section1/
│   └── ... (files related to Section 1)
├── section3/
│   ├── pii_redactor.py
│   ├── detect_pii.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   ├── GovTech RAI - Section 3 Q2.csv
│   ├── dataset_with_pii_flags.csv
│   └── README.md
```


- **`pii_redactor.py`**: Contains the `PIIRedactor` class for PII detection and redaction.
- **`detect_pii.py`**: Script to process the dataset and detect PII.
- **`requirements.txt`**: Lists the Python dependencies.
- **`Dockerfile`**: Dockerfile for building the FastAPI backend image.
- **`Dockerfile.streamlit`**: Dockerfile for building the Streamlit frontend image.
- **Dataset Files**:
  - `GovTech RAI - Section 3 Q2.csv`: The dataset containing text entries.
  - `dataset_with_pii_flags.csv`: The updated dataset with PII detection results.
- **`README.md`**: Project documentation.

---

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- **Python 3.7+**
- **pip** (Python package manager)
- **Git** (to clone the repository)
- **Docker** (if you plan to use Docker)

### Cloning the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/ahmadmirza17/Responsible-AI-Take-Home-Assessment.git
```

### Installing Dependencies

Navigate to the project directory:

```bash
cd Responsible-AI-Take-Home-Assessment/section3
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the PII Detection Script

**Script Overview**

The script `detect_pii.py` processes a dataset containing text entries, detects the presence of PII in each text, and adds new binary features indicating the presence (1) or absence (0) of each type of PII.

**Script**: `detect_pii.py`

```python
# detect_pii.py

import pandas as pd
from pii_redactor import PIIRedactor
from tqdm import tqdm

def main():
    # Initialize the PII Redactor
    redactor = PIIRedactor()

    # Load the dataset
    df = pd.read_csv(r'GovTech RAI - Section 3 Q2.csv')  # Ensure the file is in the same directory

    # Initialize lists to store PII detection results
    name_detected = []
    email_detected = []
    phone_detected = []
    nric_detected = []
    location_detected = []

    # Iterate over each text entry
    for text in tqdm(df['text'], desc="Processing texts"):
        # Detect PII in the text
        pii_results = redactor.detect_pii(text)
        
        # Append results to the lists
        name_detected.append(int(pii_results['name']))
        email_detected.append(int(pii_results['email']))
        phone_detected.append(int(pii_results['phone']))
        nric_detected.append(int(pii_results['nric']))
        location_detected.append(int(pii_results['location']))

    # Add new columns to the DataFrame
    df['name_detected'] = name_detected
    df['email_detected'] = email_detected
    df['phone_detected'] = phone_detected
    df['nric_detected'] = nric_detected
    df['location_detected'] = location_detected

    # Save the updated DataFrame to a new CSV file
    df.to_csv('dataset_with_pii_flags.csv', index=False)
    print("Processing complete. The updated dataset is saved as 'dataset_with_pii_flags.csv'.")

if __name__ == "__main__":
    main()
```

### Handling File Paths on Windows

**Issue:**

You might encounter a `SyntaxError` due to backslashes in the file path being interpreted as escape characters.

**Solution:**

- Since the dataset is in the same directory as the script, you can simply use the file name:

```python
df = pd.read_csv('GovTech RAI - Section 3 Q2.csv')
```

- If you need to specify a full path, use one of the following methods:

- **Using Raw String:**

```python
df = pd.read_csv(r'C:\Users\ahmad\Documents\GitHub\Responsible-AI-Take-Home-Assessment\section3\GovTech RAI - Section 3 Q2.csv')
```

**Using Forward Slashes:**

```python
df = pd.read_csv('C:/Users/ahmad/Documents/GitHub/Responsible-AI-Take-Home-Assessment/section3/GovTech RAI - Section 3 Q2.csv')
```

**Escaping Backslashes:**

```python
df = pd.read_csv('C:\\Users\\ahmad\\Documents\\GitHub\\Responsible-AI-Take-Home-Assessment\\section3\\GovTech RAI - Section 3 Q2.csv')
```

### Running the Script

1. Ensure your dataset CSV file is accessible and the file path in the script is correct.

2. Run the script:

```bash
python detect_pii.py
```

3. The script will process each text entry, detect PII, and save the updated dataset as `dataset_with_pii_flags.csv`.

## Assessing Tool Performance

**Suggested Performance Metrics**

When evaluating the PII detection tool, consider the following metrics:

**Precision and Recall**

- Precision: The proportion of correctly identified PII instances out of all instances flagged as PII by the tool.
 
- Recall: The proportion of actual PII instances that were correctly identified by the tool.
 
**F1 Score**

- The harmonic mean of precision and recall.

**Evaluating the Tool**

1. Create a Ground Truth Dataset:

    - Manually annotate a subset of your dataset to indicate the actual presence of each PII type.

2. Compare Predictions with Ground Truth:

    - Use the detection results from your tool and compare them against the ground truth labels.

3. Calculate Metrics:

    - For each PII type, compute precision, recall, and F1 score.

4. Analyze Results:

    - Identify areas where the tool performs well and where improvements are needed.

    - Consider the diversity and representativeness of your dataset.

## Including the Updated Dataset

Include the updated dataset `dataset_with_pii_flags.csv` in your repository under the `section3` directory.

**Directory Structure Example:**

```plaintext
Responsible-AI-Take-Home-Assessment/
├── section1/
│   └── ... (files related to Section 1)
├── section3/
│   ├── pii_redactor.py
│   ├── detect_pii.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   ├── GovTech RAI - Section 3 Q2.csv
│   ├── dataset_with_pii_flags.csv
│   └── README.md
```

## PII Detection on Dataset

I have provided a script `detect_pii.py` to detect PII in a dataset.

**To run the script:**

```bash
python detect_pii.py
```

**Script Description:**

- Loads the dataset from `GovTech RAI - Section 3 Q2.csv`.

- Detects the presence of the following PII types in each text:

    1. Name

    2. Email

    3. Phone Number

    4. NRIC

    5. Location

- Adds new binary columns to indicate the presence (1) or absence (0) of each PII type.

- Saves the updated dataset as `dataset_with_pii_flags.csv`.

## Cloud Architecture

### Diagram

![Architecture](image\image.jpg)

### Diagram Components

- Users: Represents end-users accessing the application.

- Internet: Depicts internet connectivity.

- Route 53: AWS DNS service directing traffic to the ALB.

- Application Load Balancer (ALB): Distributes incoming application traffic across multiple targets.

- Frontend ASG (Public Subnet):

  - Streamlit Instance: Runs the Streamlit frontend application.

  - Auto Scaling: Automatically adjusts the number of instances based on demand.

- Backend ASG (Private Subnet):

  - FastAPI Instance: Runs the FastAPI backend application.

  - Auto Scaling: Automatically adjusts the number of instances based on demand.

- Amazon S3: Object storage service used by backend instances.

- Amazon RDS: Relational Database Service for persistent storage.

- NAT Gateway: Allows instances in private subnets to connect to the internet for updates.

- Internet Gateway: Provides internet access to the VPC.

### Design Considerations and Decision-Making Process

**Security as a Priority**

- Protecting PII Data:

  - Backend instances handling PII are placed in private subnets, inaccessible from the internet.

  - Communication between frontend and backend is secured through VPC and security groups.

- Encryption:

  - Use HTTPS for all communications.

  - Encrypt data at rest in databases and S3.

**Rationale**: Ensuring data security and compliance with data protection regulations is critical when handling sensitive information.

**Scalability and High Availability**

- Auto Scaling Groups:

  - Enable automatic scaling based on demand patterns.

- Multi-AZ Deployment:

  - Distribute resources across multiple Availability Zones to prevent downtime due to AZ failures.

**Rationale**: Enhances user experience by maintaining performance during peak loads and ensuring service availability.

**Performance Optimisation**

- Load Balancing:

  - ALB efficiently distributes traffic, reducing bottlenecks.

- Efficient Networking:

  - Keeping communication within the VPC reduces latency.

- Caching (Optional):

  - Implement caching mechanisms if necessary to reduce load times.

**Rationale**: Optimises resource usage and improves response times for end-users.

**Cost Eficiency**

- Right-Sizing Instances:

  - Choose appropriate instance types based on workload requirements.

- Auto Scaling:

  - Scales down resources during low traffic periods to reduce costs.

- Reserved Instances or Savings Plans (Optional):

  - Commit to longer-term usage for cost savings.

**Rationale**: Balances performance needs with budget constraints.

**Deployment Strategy**

- Containerization:

  - Use Docker to containerize applications.

  - Deploy using AWS ECS or EKS for better resource utilization and management.

Rationale: Containers provide consistency across environments and facilitate scalability.

### Final Considerations

1. Leveraging More Compute Power

**Utilize Larger Language Models (LLMs)**

  - Advanced Models: Implement state-of-the-art models like GPT-3, GPT-4, or BERT to improve accuracy in PII detection.

  - Contextual Understanding: Larger models better grasp context and nuances in language, reducing false positives and negatives.

  - Complex Pattern Recognition: Enhanced ability to detect non-standard PII formats and obfuscated information.

**Advanced Computational Techniques**

  - Hardware Acceleration: Use GPUs and TPUs to accelerate training and inference.

  - Parallel Processing: Employ distributed computing to handle large datasets efficiently.

  - Scalability: Handle higher data volumes and support real-time processing.

2. Leveraging More Expertise

**Expert Data Scientists and NLP Specialists**

  - Model Fine-Tuning: Customize models on domain-specific datasets for improved performance.

  - Custom Algorithms: Develop specialized detection methods tailored to specific PII types and industries.

  - Error Analysis: Identify and address model weaknesses through detailed analysis.

**Cross-Functional Collaboration**

  - Legal and Compliance Experts: Ensure adherence to data protection regulations.

  - Security Professionals: Enhance data security measures and implement best practices.

  - UX Designers: Improve the user interface and overall user experience.

**Research and Innovation**

  - Stay Updated: Integrate the latest NLP research and advancements.

  - Experimentation: Explore new techniques like transfer learning and zero-shot learning.

3. Allocating More Time

**Extensive Model Training and Validation**

  - Longer Training Periods: Achieve better model convergence and accuracy.

  - Hyperparameter Optimization: Fine-tune settings for optimal model performance.

  - Iterative Development: Continuously improve the tool based on feedback.

**Comprehensive Testing**

  - Thorough Testing: Implement unit, integration, and user acceptance testing.

  - Data Collection and Preparation: Gather larger and more diverse datasets for training and evaluation.

  - Refinement of Detection Rules: Improve rule-based methods to complement machine learning approaches.