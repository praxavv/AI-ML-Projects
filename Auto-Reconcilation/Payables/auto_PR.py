import pandas as pd
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import os

# Set up the path to the script's directory (e.g., Payables folder)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input file paths (in same folder as script)
vendor_path = os.path.join(script_dir, 'vendor_invoices.csv')
payments_path = os.path.join(script_dir, 'outgoing_payments.csv')

# Load input data
vendor_invoices = pd.read_csv(vendor_path)
outgoing_payments = pd.read_csv(payments_path)

# Result containers
matched = []
unmatched = []

# Match each vendor invoice with appropriate outgoing payments
for _, invoice in vendor_invoices.iterrows():
    remaining_amount = invoice['Amount']
    invoice_matches = []

    for _, payment in outgoing_payments.iterrows():
        desc_score = fuzz.token_set_ratio(invoice['Description'], payment['Description'])
        name_score = fuzz.token_set_ratio(invoice['VendorName'], payment['PaidTo'])
        combined_score = (0.6 * desc_score) + (0.4 * name_score)

        if (
            combined_score > 60 and
            payment['Amount'] <= remaining_amount and
            payment['PaymentID'] not in [m['MatchedWithPaymentID'] for m in matched]
        ):
            invoice_matches.append({
                'InvoiceNo': invoice['InvoiceNo'],
                'MatchedWithPaymentID': payment['PaymentID'],
                'VendorName': invoice['VendorName'],
                'MatchedAmount': payment['Amount'],
                'MatchScore': round(combined_score, 3)
            })
            remaining_amount -= payment['Amount']

            if remaining_amount == 0:
                break

    if invoice_matches:
        matched.extend(invoice_matches)
        if remaining_amount > 0:
            unmatched.append({
                'InvoiceNo': invoice['InvoiceNo'],
                'VendorName': invoice['VendorName'],
                'RemainingAmount': remaining_amount
            })
    else:
        unmatched.append({
            'InvoiceNo': invoice['InvoiceNo'],
            'VendorName': invoice['VendorName'],
            'RemainingAmount': invoice['Amount']
        })

# Convert matches and unmatched results to DataFrames
matched_df = pd.DataFrame(matched)
unmatched_df = pd.DataFrame(unmatched)

# Classify payment status per invoice
status_labels = []
for invoice in vendor_invoices.itertuples():
    matched_rows = matched_df[matched_df['InvoiceNo'] == invoice.InvoiceNo]
    total_paid = matched_rows['MatchedAmount'].sum() if not matched_rows.empty else 0

    if total_paid == invoice.Amount:
        status_labels.append('Fully Paid')
    elif total_paid > 0:
        status_labels.append('Partially Paid')
    else:
        status_labels.append('Unpaid')

vendor_invoices['Status'] = status_labels

# Visualization of payment status
status_counts = vendor_invoices['Status'].value_counts()
plt.figure(figsize=(6, 4))
status_counts.plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Vendor Invoice Payment Status')
plt.xlabel('Status')
plt.ylabel('Number of Invoices')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 🔁 PRO TIP: Save to file in script's directory
def save_to_payables(filename, df):
    filepath = os.path.join(script_dir, filename)
    df.to_csv(filepath, index=False)

# Save output files in the same directory as script
save_to_payables('matched_payables.csv', matched_df)
save_to_payables('unpaid_vendor_invoices.csv', unmatched_df)

print("✅ Payables matching complete! Files saved to your Payables folder.")