import pandas as pd
from pii_redactor import PIIRedactor
from tqdm import tqdm

def main():
    redactor = PIIRedactor()

    df = pd.read_csv(r'C:\Users\ahmad\Documents\GitHub\Responsible-AI-Take-Home-Assessment\section3\dataset.csv')

    name_detected = []
    email_detected = []
    phone_detected = []
    nric_detected = []
    location_detected = []

    for text in tqdm(df['text'], desc="Processing texts"):
        pii_results = redactor.detect_pii(text)
        
        name_detected.append(int(pii_results['name']))
        email_detected.append(int(pii_results['email']))
        phone_detected.append(int(pii_results['phone']))
        nric_detected.append(int(pii_results['nric']))
        location_detected.append(int(pii_results['location']))

    df['name_detected'] = name_detected
    df['email_detected'] = email_detected
    df['phone_detected'] = phone_detected
    df['nric_detected'] = nric_detected
    df['location_detected'] = location_detected

    df.to_csv('dataset_with_pii_flags.csv', index=False)
    print("Processing complete. The updated dataset is saved as 'dataset_with_pii_flags.csv'.")

if __name__ == "__main__":
    main()
