import pandas as pd
from app import (
    classify_columns,
    analyze_data_consistency,
    detect_outliers_iqr,
    apply_user_controlled_preprocessing,
    generate_preprocessing_report
)

df = pd.read_csv('messy_student.csv')
print('=== 1. Loaded messy_student.csv ===')
print(df)

clf = classify_columns(df)
print('\n=== 2. Classification ===')
print(clf[['Column', 'Role', 'Reason']])

cons = analyze_data_consistency(df, clf)
print('\n=== 3. Consistency Analysis ===')
print(cons)

outs = detect_outliers_iqr(df)
print('\n=== 4. Outliers ===')
print(outs)

config = {
    'handle_identifiers': True,
    'drop_identifiers': False,
    'remove_duplicates': True,
    'standardize_categories': True,
    'category_mappings': {'City': {'chennai': 'Chennai', 'Chen nai': 'Chennai'}},
    'unconfirmed_inconsistencies': [],
    'impute_numeric': True,
    'numeric_strategy': 'Median',
    'numeric_cols': ['Age', 'CGPA', 'Salary'],
    'impute_categorical': True,
    'categorical_cols': ['Gender'],
    'outlier_action': 'Keep outliers as-is (Recommended)',
    'outlier_cols': ['Salary'],
    'scale_numeric': True,
    'scaler_type': 'StandardScaler',
    'scale_cols': ['Age', 'CGPA', 'Salary'],
    'encode_categories': True,
    'encode_drop_first': False,
    'encode_cols': ['Gender', 'City']
}

cleaned, log, unres, err = apply_user_controlled_preprocessing(df, clf, config)

print('\n=== 5. Cleaned Dataset ===')
print(cleaned)

print('\n=== 6. Applied Log ===')
for l in log:
    print(f"- {l['Action']} ({l['Method']}) on [{l['Affected Columns']}]: {l['Details']}")

print('\n=== 7. Unresolved Issues ===')
for u in unres:
    print(f"- {u}")

print('\n=== 8. Errors / Warnings ===')
print(err if err else 'None (0 errors)')

report = generate_preprocessing_report(df, cleaned, log, unres, err)
print('\n=== 9. Generated Report Preview ===')
print(report)
