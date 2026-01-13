import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

def generate_batch(num_samples, is_historical):
    departments = ['IT', 'Marketing', 'R&D', 'Operations', 'HR', 'Finance']
    
    data = {
        'Project_ID': [f'PROJ_{np.random.randint(1000, 9999)}' for _ in range(num_samples)],
        'Department': np.random.choice(departments, num_samples),
        'Investment_Capital': np.random.randint(500000, 5000000, num_samples),
        'Duration_Months': np.random.randint(6, 36, num_samples),
        'Risk_Score': np.round(np.random.uniform(1, 10, num_samples), 2),
        'Strategic_Alignment': np.random.randint(1, 6, num_samples),
        'Market_Trend_Index': np.round(np.random.uniform(0.5, 2.0, num_samples), 2),
        'Type': 'History' if is_historical else 'Proposal'
    }
    
    df = pd.DataFrame(data)
    
    # Generate Targets
    base_roi = (df['Risk_Score'] * 1.5) + (df['Market_Trend_Index'] * 5)
    noise = np.random.normal(0, 2, num_samples)
    
    if is_historical:
        df['Actual_ROI_Pct'] = np.round(base_roi + noise, 2)
        df['Actual_NPV'] = np.round((df['Investment_Capital'] * (df['Actual_ROI_Pct']/100)) / 1.1, 2)
    else:
        # For proposals, targets are unknown (NaN)
        df['Actual_ROI_Pct'] = np.nan
        df['Actual_NPV'] = np.nan
        
    return df

# Generate and Combine
df_hist = generate_batch(500, is_historical=True)
df_prop = generate_batch(50, is_historical=False)

df_master = pd.concat([df_hist, df_prop], ignore_index=True)
df_master.to_csv("master_project_data.csv", index=False)

print(f"✅ Created 'master_project_data.csv' with {len(df_master)} rows.")
print("Upload this single file to your app.")
