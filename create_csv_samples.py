"""
Convert WFDB files to CSV for easy web upload
"""
import wfdb
import numpy as np
import os

data_dir = 'data'
output_dir = 'sample_ecg_files'

os.makedirs(output_dir, exist_ok=True)

# Convert a few sample files
samples = ['a01', 'a02', 'a03', 'a05', 'a06', 'a07']

print("Converting WFDB files to CSV...")
print("="*60)

for record_name in samples:
    try:
        # Read WFDB record
        record = wfdb.rdrecord(f'{data_dir}/{record_name}')
        ecg = record.p_signal[:, 0]
        
        # Save as CSV (first 5 minutes for faster upload)
        output_file = f'{output_dir}/{record_name}_ecg.csv'
        np.savetxt(output_file, ecg[:30000], delimiter=',', fmt='%.6f')
        
        duration_min = len(ecg[:30000]) / 100 / 60
        print(f"✓ {record_name}_ecg.csv - {len(ecg[:30000]):,} samples ({duration_min:.1f} min)")
        
    except Exception as e:
        print(f"✗ Error with {record_name}: {e}")

print("="*60)
print(f"\n✅ Done! CSV files created in '{output_dir}' folder")
print("\nYou can now upload these .csv files to the web app!")
print("Example: a01_ecg.csv, a02_ecg.csv, etc.")
