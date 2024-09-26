import pandas as pd
from pii_redactor import PIIRedactor
from tqdm import tqdm

def main():
    # Initialize the PII Redactor
    redactor = PIIRedactor()

    # Load the dataset
    df = pd.read_csv(r'C:\Users\ahmad\Documents\GitHub\Responsible-AI-Take-Home-Assessment\section3\dataset.csv')

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
