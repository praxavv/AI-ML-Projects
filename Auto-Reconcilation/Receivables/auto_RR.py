import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt


# Load CSVs
invoices = pd.read_csv('invoices.csv')
payments = pd.read_csv('payments.csv')

# Result lists
matched = []
unmatched = []

# Match each invoice
for _, invoice in invoices.iterrows():
    remaining_amount = invoice['Amount']
    invoice_matches = []
    
    for _, payment in payments.iterrows():
        desc_score = fuzz.token_set_ratio(invoice['Description'], payment['Description'])
        name_score = fuzz.token_set_ratio(invoice['ClientName'], payment['Payer'])

        # Weighted average (you can tweak the weights)
        combined_score = (0.6 * desc_score) + (0.4 * name_score)


        
        if (
            combined_score > 60 and
            payment['Amount'] <= remaining_amount and
            payment['PaymentID'] not in [m['MatchedWithPaymentID'] for m in matched]
        ):
            invoice_matches.append({
                'InvoiceNo': invoice['InvoiceNo'],
                'MatchedWithPaymentID': payment['PaymentID'],
                'ClientName': invoice['ClientName'],
                'MatchedAmount': payment['Amount'],
                'MatchScore': combined_score
            })
            remaining_amount -= payment['Amount']

            if remaining_amount == 0:
                break

    if invoice_matches:
        matched.extend(invoice_matches)
        if remaining_amount > 0:
            unmatched.append({
                'InvoiceNo': invoice['InvoiceNo'],
                'ClientName': invoice['ClientName'],
                'RemainingAmount': remaining_amount
            })
    else:
        unmatched.append({
            'InvoiceNo': invoice['InvoiceNo'],
            'ClientName': invoice['ClientName'],
            'RemainingAmount': invoice['Amount']
        })

# Convert to DataFrames
matched_df = pd.DataFrame(matched)
unmatched_df = pd.DataFrame(unmatched)

# Prepare data for visualization
status_labels = []
for invoice in invoices.itertuples():
    matched_rows = matched_df[matched_df['InvoiceNo'] == invoice.InvoiceNo]
    total_paid = matched_rows['MatchedAmount'].sum() if not matched_rows.empty else 0

    if total_paid == invoice.Amount:
        status_labels.append('Fully Paid')
    elif total_paid > 0:
        status_labels.append('Partially Paid')
    else:
        status_labels.append('Unpaid')

invoices['Status'] = status_labels
status_counts = invoices['Status'].value_counts()

plt.figure(figsize=(6, 4))
status_counts.plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Invoice Payment Status')
plt.xlabel('Status')
plt.ylabel('Number of Invoices')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Save output
matched_df.to_csv('matched_invoices.csv', index=False)
unmatched_df.to_csv('unpaid_invoices.csv', index=False)

print("✅ Matching complete! Check 'matched_invoices.csv' and 'unpaid_invoices.csv'")
