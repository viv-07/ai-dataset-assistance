import io
import os
import json
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai as _genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# Top-level UI moved into main()

# -------------------- Helper functions --------------------

IDENTIFIER_KEYWORDS = {
    "id", "identifier", "student_name", "student_id", "studentid",
    "customer_id", "customerid", "employee_id", "employeeid",
    "roll_no", "rollno", "registration_no", "registration_number",
    "reg_no", "email", "phone", "mobile", "name"
}


def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def profile_dataset(df):
    profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique Values": df.nunique(dropna=True).values,
    })
    return profile


def looks_like_identifier(col, series):
    """Return True when a column is likely an identifier/non-predictive field."""
    normalized = (
        str(col).strip().lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in IDENTIFIER_KEYWORDS:
        return True

    if any(
        normalized.endswith(suffix)
        for suffix in ("_id", "_name", "_email", "_phone", "_mobile", "_number")
    ):
        return True

    # Very high-cardinality text columns are often names, emails, codes, etc.
    non_null = series.dropna()
    if len(non_null) >= 5 and not pd.api.types.is_numeric_dtype(series):
        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio >= 0.95:
            return True

    return False


def classify_columns(df, target_col=None):
    """Classify columns as Target, Identifier, Numerical, or Categorical."""
    rows = []

    for col in df.columns:
        series = df[col]

        if target_col and col == target_col:
            role = "Target"
            reason = "Selected by the user as the prediction target."
        elif looks_like_identifier(col, series):
            role = "Identifier / Non-predictive"
            reason = "Likely identifies a record rather than representing a useful predictive feature."
        elif pd.api.types.is_numeric_dtype(series):
            role = "Numerical"
            reason = "Contains numeric values suitable for numerical analysis."
        else:
            role = "Categorical"
            reason = "Contains discrete text/category values that may need encoding for ML."

        rows.append({
            "Column": col,
            "Role": role,
            "Data Type": str(series.dtype),
            "Unique Values": int(series.nunique(dropna=True)),
            "Reason": reason,
        })

    return pd.DataFrame(rows)


def analyze_data_consistency(df, classification_df):
    """
    Detect potentially inconsistent category values in Categorical columns.
    Normalizes values ONLY for comparison (casefold, whitespace stripping, collapsing spaces).
    Distinguishes between clear formatting differences and possible spelling/spacing differences.
    """
    results = []

    categorical_cols = classification_df.loc[
        classification_df["Role"] == "Categorical", "Column"
    ].tolist()

    for col in categorical_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        value_counts = series.astype(str).value_counts()
        unique_vals = list(value_counts.index)

        if len(unique_vals) <= 1:
            continue

        norm_clean_map = {}
        norm_nospace_map = {}
        nospace_groups = {}

        for val in unique_vals:
            val_str = str(val)
            # Normalization: Strip leading/trailing whitespace, convert to casefold, collapse repeated internal spaces
            clean = " ".join(val_str.strip().casefold().split())
            # Secondary normalization: Remove spaces and non-alphanumeric characters for spacing/spelling variant matching
            nospace = "".join(c for c in clean if c.isalnum())

            norm_clean_map[val_str] = clean
            norm_nospace_map[val_str] = nospace

            if nospace not in nospace_groups:
                nospace_groups[nospace] = []
            nospace_groups[nospace].append(val_str)

        for nospace_key, group in nospace_groups.items():
            if len(group) > 1:
                # Sort variants by frequency in the dataset (descending), then alphabetically
                sorted_group = sorted(group, key=lambda v: (-value_counts[v], v))
                suggested_standard = sorted_group[0]
                std_clean = norm_clean_map[suggested_standard]

                # Group raw values by clean representation to separate formatting vs spacing differences
                clean_subgroups = {}
                for v in group:
                    c = norm_clean_map[v]
                    if c not in clean_subgroups:
                        clean_subgroups[c] = []
                    clean_subgroups[c].append(v)

                formatting_diffs = []
                spelling_spacing_diffs = []

                for c_key, c_vals in clean_subgroups.items():
                    if c_key == std_clean:
                        for v in c_vals:
                            if v != suggested_standard:
                                formatting_diffs.append(v)
                    else:
                        for v in c_vals:
                            spelling_spacing_diffs.append(v)

                fmt_verb = "is a" if len(formatting_diffs) == 1 else "are"
                fmt_noun = "clear formatting difference" if len(formatting_diffs) == 1 else "clear formatting differences"
                sp_verb = "is a" if len(spelling_spacing_diffs) == 1 else "are"
                sp_noun = "possible spelling/spacing difference" if len(spelling_spacing_diffs) == 1 else "possible spelling/spacing differences"

                if formatting_diffs and spelling_spacing_diffs:
                    fmt_str = ", ".join([f"'{v}'" for v in formatting_diffs])
                    sp_str = ", ".join([f"'{v}'" for v in spelling_spacing_diffs])
                    reason_text = (
                        f"{fmt_str} {fmt_verb} {fmt_noun} (casing/whitespace), and "
                        f"{sp_str} {sp_verb} {sp_noun} compared to '{suggested_standard}'."
                    )
                elif formatting_diffs:
                    fmt_str = ", ".join([f"'{v}'" for v in formatting_diffs])
                    reason_text = (
                        f"{fmt_str} {fmt_verb} {fmt_noun} (casing/whitespace) compared to '{suggested_standard}'."
                    )
                elif spelling_spacing_diffs:
                    sp_str = ", ".join([f"'{v}'" for v in spelling_spacing_diffs])
                    reason_text = (
                        f"{sp_str} {sp_verb} {sp_noun} compared to '{suggested_standard}'."
                    )
                else:
                    reason_text = f"Inconsistent representations of '{suggested_standard}'."

                results.append({
                    "Column": col,
                    "Detected Variants": ", ".join(group),
                    "Suggested Standard Value": suggested_standard,
                    "Reason": reason_text
                })

    return pd.DataFrame(results)


def detect_outliers_iqr(df, excluded_cols=None):
    results = []
    excluded_cols = set(excluded_cols or [])
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col in excluded_cols:
            continue

        series = df[col].dropna()
        if len(series) < 4:
            count = 0
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())

        results.append({"Column": col, "Outliers (IQR)": count})

    return pd.DataFrame(results)


def make_recommendations(df, classification_df, outlier_df=None, consistency_df=None):
    """
    Context-aware preprocessing recommendation engine combining:
    - Smart Column Classification
    - Missing-value analysis
    - Duplicate analysis
    - IQR outlier analysis
    - Data Consistency Analysis
    Combines multiple issues for the same column into a single meaningful recommendation.
    """
    recommendations = []
    roles = dict(zip(classification_df["Column"], classification_df["Role"]))

    outlier_map = {}
    if outlier_df is not None and not outlier_df.empty:
        for _, row in outlier_df.iterrows():
            outlier_map[row["Column"]] = row.get("Outliers (IQR)", 0)

    consistency_map = {}
    if consistency_df is not None and not consistency_df.empty:
        for _, row in consistency_df.iterrows():
            consistency_map[row["Column"]] = row.get("Detected Variants", "")

    for col in df.columns:
        role = roles.get(col, "")
        missing_pct = df[col].isna().mean() * 100
        has_missing = missing_pct > 0
        outliers_count = outlier_map.get(col, 0)
        has_outliers = outliers_count > 0
        variants_str = consistency_map.get(col, "")
        has_inconsistencies = bool(variants_str)

        # 1. Identifier / Non-predictive column
        if role == "Identifier / Non-predictive":
            issues = ["Identifier / non-predictive column"]
            if has_missing:
                issues.append(f"Missing values ({missing_pct:.1f}%)")
            recommendations.append({
                "Column": col,
                "Detected Issues": " | ".join(issues),
                "Recommended Action": "Exclude from ML features",
                "Reason": "Identifiers label individual records rather than providing meaningful predictive patterns for ML models.",
                "Risk/Note": "Blindly encoding high-cardinality identifiers leads to feature explosion and model overfitting."
            })
            continue

        # 2. Prediction Target column
        if role == "Target":
            issues = ["Prediction target"]
            if has_missing:
                issues.append(f"Missing values ({missing_pct:.1f}%)")
            recommendations.append({
                "Column": col,
                "Detected Issues": " | ".join(issues),
                "Recommended Action": "Isolate from input feature preprocessing",
                "Reason": "The target variable should be handled separately based on the task objective and should not be scaled or imputed with input features.",
                "Risk/Note": "Automatic target imputation or scaling can cause data leakage or alter the prediction objective."
            })
            continue

        # 3. Numerical feature column
        if role == "Numerical":
            issues = []
            if has_missing:
                issues.append(f"Missing values ({missing_pct:.1f}%)")
            if has_outliers:
                issues.append(f"{outliers_count} IQR outlier(s)")

            if has_missing and has_outliers:
                recommendations.append({
                    "Column": col,
                    "Detected Issues": " | ".join(issues),
                    "Recommended Action": "Use median imputation for missing values; review outliers before removal",
                    "Reason": "Median imputation is preferred over mean because the median is robust to extreme outlier values. Outliers should be manually inspected.",
                    "Risk/Note": "Mean imputation would be skewed by outliers. Automatically deleting outlier records without review risks losing valid extreme data points."
                })
            elif has_missing:
                recommendations.append({
                    "Column": col,
                    "Detected Issues": " | ".join(issues),
                    "Recommended Action": "Apply median or mean imputation",
                    "Reason": "Numerical algorithms require complete data matrices for computation.",
                    "Risk/Note": "Imputation modifies feature distributions. High missing rates (>50%) may indicate the column should be dropped."
                })
            elif has_outliers:
                recommendations.append({
                    "Column": col,
                    "Detected Issues": " | ".join(issues),
                    "Recommended Action": "Review outliers manually; consider winsorization or robust scaling",
                    "Reason": "Values lie beyond 1.5 * IQR boundaries. Review determines if these are measurement errors or legitimate extreme values.",
                    "Risk/Note": "Do not automatically drop outlier records, as extreme values may carry crucial predictive signal."
                })
            continue

        # 4. Categorical feature column
        if role == "Categorical":
            issues = []
            if has_missing:
                issues.append(f"Missing values ({missing_pct:.1f}%)")
            if has_inconsistencies:
                issues.append(f"Inconsistent categories ({variants_str})")

            if has_missing and has_inconsistencies:
                recommendations.append({
                    "Column": col,
                    "Detected Issues": " | ".join(issues),
                    "Recommended Action": "Standardize category labels first, then apply mode imputation and categorical encoding",
                    "Reason": "Inconsistent labels (e.g. casing/spacing variants) duplicate categories. Standardizing labels first ensures accurate mode imputation and clean encoding.",
                    "Risk/Note": "Encoding before label standardization creates redundant dummy columns (e.g., separate features for 'Chennai' and 'chennai')."
                })
            elif has_inconsistencies:
                recommendations.append({
                    "Column": col,
                    "Detected Issues": " | ".join(issues),
                    "Recommended Action": "Review and standardize category labels before encoding",
                    "Reason": "Detected category variants (formatting or spacing differences) represent identical underlying concepts.",
                    "Risk/Note": "Encoding un-standardized categories increases feature cardinality and degrades model training performance."
                })
            elif has_missing:
                recommendations.append({
                    "Column": col,
                    "Detected Issues": " | ".join(issues),
                    "Recommended Action": "Apply mode (most frequent) imputation, then encode categories",
                    "Reason": "Missing text categories are imputed using the most frequent category prior to numeric encoding.",
                    "Risk/Note": "Mode imputation assumes missing values follow the predominant category distribution."
                })
            else:
                recommendations.append({
                    "Column": col,
                    "Detected Issues": "Categorical feature",
                    "Recommended Action": "Encode categorical values (e.g., One-Hot Encoding) before ML",
                    "Reason": "Categorical text values must be converted to numerical representations for ML models.",
                    "Risk/Note": "High cardinality features may produce sparse matrices if One-Hot encoded."
                })
            continue

    # 5. Dataset Level: Duplicate Rows
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        recommendations.append({
            "Column": "Entire dataset",
            "Detected Issues": f"{duplicates} duplicate row(s)",
            "Recommended Action": "Review and remove duplicate rows",
            "Reason": "Duplicate records distort statistical summaries and risk data leakage between training and validation sets.",
            "Risk/Note": "Verify duplicates are genuine redundant entries rather than legitimate identical transactions."
        })

    return pd.DataFrame(recommendations)


def apply_preprocessing(
    df,
    classification_df,
    missing_strategy,
    encode_categories,
    scale_numeric,
    remove_duplicates,
):
    result = df.copy()
    roles = dict(zip(classification_df["Column"], classification_df["Role"]))

    if remove_duplicates:
        result = result.drop_duplicates()

    # Missing values: do not automatically modify identifiers or targets.
    for col in result.columns:
        if result[col].isna().sum() == 0:
            continue

        role = roles.get(col, "")
        if role == "Identifier / Non-predictive" or role == "Target":
            continue

        if pd.api.types.is_numeric_dtype(result[col]):
            if missing_strategy == "Median":
                result[col] = result[col].fillna(result[col].median())
            elif missing_strategy == "Mean":
                result[col] = result[col].fillna(result[col].mean())
        else:
            mode = result[col].mode(dropna=True)
            if not mode.empty:
                result[col] = result[col].fillna(mode.iloc[0])

    # Encode only genuine categorical features, excluding identifiers and target.
    if encode_categories:
        categorical_cols = [
            col for col, role in roles.items()
            if role == "Categorical" and col in result.columns
        ]
        if categorical_cols:
            result = pd.get_dummies(
                result,
                columns=categorical_cols,
                drop_first=False,
                dtype=int,
            )

    # Scale numerical features, excluding identifiers and target.
    if scale_numeric:
        numeric_cols = [
            col for col, role in roles.items()
            if role == "Numerical" and col in result.columns
            and pd.api.types.is_numeric_dtype(result[col])
        ]
        if numeric_cols:
            scaler = StandardScaler()
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])

    return result


def apply_user_controlled_preprocessing(
    df,
    classification_df,
    actions_config,
):
    """
    Apply user-selected preprocessing actions to a COPY of df.
    Validates column types, logs applied transformations, catches errors safely,
    and returns (cleaned_df, applied_actions_log, unresolved_issues_list, errors_log).
    """
    result = df.copy()
    applied_log = []
    unresolved_issues = []
    errors_log = []
    roles = dict(zip(classification_df["Column"], classification_df["Role"]))

    # 1. Identifier Columns Handling
    id_cols = [c for c, r in roles.items() if r == "Identifier / Non-predictive" and c in result.columns]
    if id_cols:
        if actions_config.get("handle_identifiers", True):
            drop_ids = actions_config.get("drop_identifiers", False)
            if drop_ids:
                try:
                    result = result.drop(columns=id_cols)
                    applied_log.append({
                        "Action": "Identifier Columns",
                        "Method": "Excluded from ML features & Dropped from dataset",
                        "Affected Columns": ", ".join(id_cols),
                        "Details": f"Dropped {len(id_cols)} non-predictive identifier column(s) to avoid feature explosion."
                    })
                except Exception as exc:
                    errors_log.append(f"Failed to drop identifier columns: {exc}")
            else:
                applied_log.append({
                    "Action": "Identifier Columns",
                    "Method": "Excluded from feature transformations (Retained as metadata)",
                    "Affected Columns": ", ".join(id_cols),
                    "Details": f"Protected {len(id_cols)} identifier column(s) from scaling and one-hot encoding."
                })
        else:
            unresolved_issues.append("Identifier columns were not explicitly managed; verify they are not encoded blindly.")

    # 2. Duplicate Rows Removal
    dup_count_initial = int(result.duplicated().sum())
    if dup_count_initial > 0:
        if actions_config.get("remove_duplicates", False):
            try:
                result = result.drop_duplicates()
                applied_log.append({
                    "Action": "Remove Duplicate Rows",
                    "Method": "drop_duplicates()",
                    "Affected Columns": "All columns (Row-level)",
                    "Details": f"Removed {dup_count_initial} duplicate row(s)."
                })
            except Exception as exc:
                errors_log.append(f"Failed to remove duplicates: {exc}")
        else:
            unresolved_issues.append(f"Duplicate rows ({dup_count_initial} row(s)) were detected but not removed.")

    # 3. Inconsistent Categorical Values (Review & Standardization)
    if actions_config.get("standardize_categories", False):
        confirmed_mappings = actions_config.get("category_mappings", {})
        for col, mapping in confirmed_mappings.items():
            if col in result.columns and mapping:
                try:
                    result[col] = result[col].replace(mapping)
                    variants_list = [f"'{k}' -> '{v}'" for k, v in mapping.items()]
                    applied_log.append({
                        "Action": "Standardize Inconsistent Categories",
                        "Method": "User-confirmed value mapping",
                        "Affected Columns": col,
                        "Details": f"Standardized {len(mapping)} value(s): {', '.join(variants_list)}."
                    })
                except Exception as exc:
                    errors_log.append(f"Failed to standardize categories in '{col}': {exc}")

    unconfirmed_inconsistencies = actions_config.get("unconfirmed_inconsistencies", [])
    for unconf in unconfirmed_inconsistencies:
        unresolved_issues.append(f"Category inconsistency in '{unconf['col']}': variant '{unconf['variants']}' was not standardized.")

    # 4. Numerical Outliers Handling
    outlier_action = actions_config.get("outlier_action", "Keep outliers as-is (Recommended)")
    outlier_cols = actions_config.get("outlier_cols", [])
    if outlier_cols:
        if "Cap outliers" in outlier_action:
            for col in outlier_cols:
                if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                    try:
                        s = result[col].dropna()
                        if len(s) >= 4:
                            q1 = float(s.quantile(0.25))
                            q3 = float(s.quantile(0.75))
                            iqr = q3 - q1
                            lower = q1 - 1.5 * iqr
                            upper = q3 + 1.5 * iqr
                            capped_count = int(((result[col] < lower) | (result[col] > upper)).sum())
                            if capped_count > 0:
                                result[col] = result[col].clip(lower=lower, upper=upper)
                                applied_log.append({
                                    "Action": "Numerical Outlier Capping",
                                    "Method": f"IQR Winsorization ([{lower:.2f}, {upper:.2f}])",
                                    "Affected Columns": col,
                                    "Details": f"Capped {capped_count} outlier value(s) within 1.5*IQR bounds."
                                })
                    except Exception as exc:
                        errors_log.append(f"Failed to cap outliers in '{col}': {exc}")
        elif "Remove rows" in outlier_action:
            for col in outlier_cols:
                if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                    try:
                        s = result[col].dropna()
                        if len(s) >= 4:
                            q1 = float(s.quantile(0.25))
                            q3 = float(s.quantile(0.75))
                            iqr = q3 - q1
                            lower = q1 - 1.5 * iqr
                            upper = q3 + 1.5 * iqr
                            mask = (result[col].isna()) | ((result[col] >= lower) & (result[col] <= upper))
                            dropped_count = len(result) - int(mask.sum())
                            if dropped_count > 0:
                                result = result[mask].copy()
                                applied_log.append({
                                    "Action": "Remove Outlier Rows",
                                    "Method": "Row deletion outside 1.5*IQR",
                                    "Affected Columns": col,
                                    "Details": f"Removed {dropped_count} row(s) containing extreme outliers."
                                })
                    except Exception as exc:
                        errors_log.append(f"Failed to remove outlier rows in '{col}': {exc}")
        else: # Keep as-is
            applied_log.append({
                "Action": "Numerical Outliers Review",
                "Method": "Preserved without modification",
                "Affected Columns": ", ".join(outlier_cols),
                "Details": "Outliers kept intact to preserve natural extreme variation and predictive signal."
            })
            unresolved_issues.append(f"Outliers in [{', '.join(outlier_cols)}] were reviewed and preserved as-is per user decision.")

    # 5. Numerical Missing Values Imputation
    if actions_config.get("impute_numeric", False):
        num_strategy = actions_config.get("numeric_strategy", "Median")
        num_cols = actions_config.get("numeric_cols", [])
        for col in num_cols:
            if col in result.columns:
                if not pd.api.types.is_numeric_dtype(result[col]):
                    errors_log.append(f"Cannot perform numeric imputation on non-numeric column '{col}'.")
                    continue
                missing_cnt = int(result[col].isna().sum())
                if missing_cnt > 0:
                    try:
                        if "Mean" in num_strategy:
                            fill_val = float(result[col].mean())
                        else:
                            fill_val = float(result[col].median())
                        result[col] = result[col].fillna(fill_val)
                        applied_log.append({
                            "Action": "Numerical Missing Value Imputation",
                            "Method": f"{num_strategy} Imputation (fill value: {fill_val:.4f})",
                            "Affected Columns": col,
                            "Details": f"Imputed {missing_cnt} missing cell(s)."
                        })
                    except Exception as exc:
                        errors_log.append(f"Failed to impute numerical column '{col}': {exc}")

    # 6. Categorical Missing Values Imputation (Mode)
    if actions_config.get("impute_categorical", False):
        cat_cols = actions_config.get("categorical_cols", [])
        for col in cat_cols:
            if col in result.columns:
                missing_cnt = int(result[col].isna().sum())
                if missing_cnt > 0:
                    try:
                        modes = result[col].mode(dropna=True)
                        if not modes.empty:
                            fill_val = modes.iloc[0]
                            result[col] = result[col].fillna(fill_val)
                            applied_log.append({
                                "Action": "Categorical Missing Value Imputation",
                                "Method": f"Mode Imputation (fill value: '{fill_val}')",
                                "Affected Columns": col,
                                "Details": f"Imputed {missing_cnt} missing cell(s)."
                            })
                        else:
                            errors_log.append(f"Column '{col}' has no non-null mode to impute.")
                    except Exception as exc:
                        errors_log.append(f"Failed to impute categorical column '{col}': {exc}")

    # Check for remaining missing values in any columns
    remaining_missing_total = int(result.isna().sum().sum())
    if remaining_missing_total > 0:
        cols_with_missing = [c for c in result.columns if result[c].isna().sum() > 0]
        unresolved_issues.append(
            f"Missing values remain in {len(cols_with_missing)} column(s): {', '.join([f'{c} ({int(result[c].isna().sum())})' for c in cols_with_missing])}."
        )

    # 7. Numerical Feature Scaling
    if actions_config.get("scale_numeric", False):
        scaler_type = actions_config.get("scaler_type", "StandardScaler")
        scale_cols = actions_config.get("scale_cols", [])
        valid_scale_cols = []
        for col in scale_cols:
            if col not in result.columns:
                continue
            if not pd.api.types.is_numeric_dtype(result[col]):
                errors_log.append(f"Cannot scale non-numeric column '{col}'.")
                continue
            if result[col].isna().sum() > 0:
                errors_log.append(f"Cannot scale column '{col}' because it contains missing values. Impute missing values first.")
                continue
            valid_scale_cols.append(col)

        if valid_scale_cols:
            try:
                if "MinMax" in scaler_type:
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                result[valid_scale_cols] = scaler.fit_transform(result[valid_scale_cols])
                applied_log.append({
                    "Action": "Numerical Feature Scaling",
                    "Method": f"{scaler_type}",
                    "Affected Columns": ", ".join(valid_scale_cols),
                    "Details": f"Rescaled {len(valid_scale_cols)} numeric feature column(s)."
                })
            except Exception as exc:
                errors_log.append(f"Failed to scale numerical columns: {exc}")

    # 8. Categorical Feature Encoding (One-Hot)
    if actions_config.get("encode_categories", False):
        encode_cols = actions_config.get("encode_cols", [])
        drop_first = actions_config.get("encode_drop_first", False)
        valid_encode_cols = [
            c for c in encode_cols
            if c in result.columns and roles.get(c) not in ("Identifier / Non-predictive", "Target")
        ]
        if valid_encode_cols:
            try:
                cols_before = result.shape[1]
                result = pd.get_dummies(
                    result,
                    columns=valid_encode_cols,
                    drop_first=drop_first,
                    dtype=int
                )
                cols_after = result.shape[1]
                applied_log.append({
                    "Action": "Categorical Feature Encoding",
                    "Method": f"One-Hot Encoding (drop_first={drop_first})",
                    "Affected Columns": ", ".join(valid_encode_cols),
                    "Details": f"Transformed {len(valid_encode_cols)} categorical column(s) into {cols_after - cols_before + len(valid_encode_cols)} binary features."
                })
            except Exception as exc:
                errors_log.append(f"Failed to one-hot encode categories: {exc}")

    return result, applied_log, unresolved_issues, errors_log


def generate_preprocessing_report(
    original_df,
    cleaned_df,
    applied_actions,
    unresolved_issues,
    errors_log=None
):
    """
    Generate a comprehensive Markdown report documenting before/after metrics,
    transformations applied, columns affected, and any remaining unresolved issues.
    """
    report_lines = [
        "# Data Preprocessing & Cleaning Report",
        "",
        "## 1. Executive Summary",
        f"- **Original Dimensions**: {original_df.shape[0]} rows x {original_df.shape[1]} columns",
        f"- **Cleaned Dimensions**: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns",
        f"- **Row Change**: {cleaned_df.shape[0] - original_df.shape[0]} rows",
        f"- **Column Change**: {cleaned_df.shape[1] - original_df.shape[1]} columns",
        f"- **Missing Values Before**: {int(original_df.isna().sum().sum())} cells",
        f"- **Missing Values After**: {int(cleaned_df.isna().sum().sum())} cells",
        f"- **Duplicate Rows Before**: {int(original_df.duplicated().sum())} rows",
        f"- **Duplicate Rows After**: {int(cleaned_df.duplicated().sum())} rows",
        "",
        "## 2. Before vs. After Comparison Table",
        "| Metric | Original Dataset | Cleaned Dataset | Change |",
        "| :--- | :--- | :--- | :--- |",
        f"| Rows | {original_df.shape[0]} | {cleaned_df.shape[0]} | {cleaned_df.shape[0] - original_df.shape[0]} |",
        f"| Columns | {original_df.shape[1]} | {cleaned_df.shape[1]} | {cleaned_df.shape[1] - original_df.shape[1]} |",
        f"| Missing Cells | {int(original_df.isna().sum().sum())} | {int(cleaned_df.isna().sum().sum())} | {int(cleaned_df.isna().sum().sum()) - int(original_df.isna().sum().sum())} |",
        f"| Duplicate Rows | {int(original_df.duplicated().sum())} | {int(cleaned_df.duplicated().sum())} | {int(cleaned_df.duplicated().sum()) - int(original_df.duplicated().sum())} |",
        "",
        "## 3. Applied Preprocessing Actions",
    ]

    if applied_actions:
        report_lines.append("| Action | Method | Affected Columns | Details |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        for act in applied_actions:
            report_lines.append(
                f"| {act.get('Action', '')} | {act.get('Method', '')} | {act.get('Affected Columns', '')} | {act.get('Details', '')} |"
            )
    else:
        report_lines.append("_No preprocessing transformations were applied._")

    report_lines.extend([
        "",
        "## 4. Unresolved Issues & Observations",
    ])

    if unresolved_issues:
        for item in unresolved_issues:
            report_lines.append(f"- [UNRESOLVED] {item}")
    else:
        report_lines.append("- [RESOLVED] All detected data quality and consistency issues were addressed.")

    if errors_log:
        report_lines.extend([
            "",
            "## 5. Processing Warnings / Errors",
        ])
        for err in errors_log:
            report_lines.append(f"- [ERROR] {err}")

    report_lines.extend([
        "",
        "---",
        "_Report generated by AI-Powered Dataset Understanding & Data Preprocessing Assistant_"
    ])

    return "\n".join(report_lines)


# -------------------- AI Explanation helpers --------------------

def build_column_summary(col, role, df, outlier_map, consistency_map):
    """
    Build a small structured dictionary with key statistics for a single column
    to be sent to the AI for explanation. Raw data is NOT included.
    """
    series = df[col]
    summary = {
        "column": col,
        "role": role,
        "dtype": str(series.dtype),
        "total_rows": len(series),
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(series.isna().mean() * 100, 1),
        "outlier_count": outlier_map.get(col, 0),
        "consistency_variants": consistency_map.get(col, ""),
    }
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            summary["mean"] = round(float(non_null.mean()), 4)
            summary["median"] = round(float(non_null.median()), 4)
            summary["std"] = round(float(non_null.std()), 4)
            summary["min"] = round(float(non_null.min()), 4)
            summary["max"] = round(float(non_null.max()), 4)
    else:
        vc = series.dropna().astype(str).value_counts()
        summary["unique_values"] = int(series.nunique(dropna=True))
        summary["top_categories"] = vc.head(5).to_dict()
    return summary


def get_ai_explanation(api_key, rec_row, col_summary, preferred_model=None):
    """
    Send a structured summary (no raw data) to the Gemini API and return
    a plain-English explanation of the recommendation.
    Uses the google-genai SDK (not the deprecated google-generativeai package).
    Returns a string explanation, or None when api_key is empty.
    """
    if not _GEMINI_AVAILABLE:
        return "google-genai package is not installed. Run: pip install google-genai"
    if not api_key:
        return None  # Caller handles the missing-key UI message.

    prompt = (
        "You are an expert data scientist explaining a preprocessing recommendation to a non-expert user.\n"
        "You are given a structured summary of a single dataset column and a preprocessing recommendation.\n"
        "Do NOT modify the dataset. Your task is only to explain.\n\n"
        f"Column summary (JSON):\n{json.dumps(col_summary, indent=2)}\n\n"
        f"Detected issues: {rec_row.get('Detected Issues', '')}\n"
        f"Recommended action: {rec_row.get('Recommended Action', '')}\n"
        f"Deterministic reason: {rec_row.get('Reason', '')}\n"
        f"Risk/Note: {rec_row.get('Risk/Note', '')}\n\n"
        "Explain in 3-5 clear sentences:\n"
        "1. What issue was detected in this column and why it matters.\n"
        "2. Why the recommended preprocessing method is appropriate here.\n"
        "3. Why an alternative approach may be less suitable.\n"
        "4. Any important risk or caution the user should keep in mind.\n"
        "Write directly to the user. Use plain, non-technical language where possible."
    )

    default_models = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    ]
    candidate_models = []
    if preferred_model:
        candidate_models.append(preferred_model)
    for m in default_models:
        if m not in candidate_models:
            candidate_models.append(m)

    errors = []
    try:
        client = _genai.Client(api_key=api_key)
    except Exception as exc:
        return f"Failed to initialize Gemini Client: {exc}"

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as exc:
            errors.append(f"Model '{model_name}': {exc}")

    return "AI explanation unavailable:\n" + "\n".join(errors)


# -------------------- Main Application --------------------

def main():
    st.set_page_config(page_title="AI Dataset Assistant", page_icon="📊", layout="wide")
    st.title("AI-Powered Dataset Understanding & Data Preprocessing Assistant")
    st.write("Upload a CSV or Excel dataset to automatically understand, analyze, and preprocess it.")

    # -------------------- Upload --------------------

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="CSV and Excel files are supported."
    )

    df = None
    if uploaded_file is None:
        st.info("Upload a dataset to begin.")
        st.stop()

    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        st.stop()

    # -------------------- Dataset overview --------------------

    st.header("1. Dataset Understanding")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Cells", int(df.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Profile")
    profile = profile_dataset(df)
    st.dataframe(profile, use_container_width=True)

    # -------------------- Smart column classification --------------------

    st.header("2. Smart Column Classification")
    st.write(
        "The assistant classifies columns before preprocessing so that identifiers are not blindly encoded "
        "and prediction targets are kept separate from input features."
    )

    target_options = ["None"] + list(df.columns)
    target_selection = st.selectbox(
        "Optional: select the prediction target",
        target_options,
        help="Choose a target only if you plan to build a prediction model. Otherwise leave it as None."
    )
    target_col = None if target_selection == "None" else target_selection

    classification_df = classify_columns(df, target_col=target_col)
    st.dataframe(classification_df, use_container_width=True)

    col_role_counts = classification_df["Role"].value_counts()
    summary_text = " | ".join(f"{role}: {count}" for role, count in col_role_counts.items())
    st.caption(f"Classification summary: {summary_text}")

    # -------------------- Data Consistency Analysis --------------------

    st.header("3. Data Consistency Analysis")
    st.write(
        "Detects potentially inconsistent category values in categorical columns caused by "
        "formatting differences (such as casing or whitespace) or minor spelling/spacing variations."
    )

    consistency_df = analyze_data_consistency(df, classification_df)
    if consistency_df.empty:
        st.info("No categorical consistency issues detected.")
    else:
        st.dataframe(consistency_df, use_container_width=True)

    st.info("Detected inconsistencies are suggestions for review. Original values are not modified automatically.")

    # -------------------- Statistics --------------------

    st.header("4. Data Quality Analysis")

    st.subheader("Numerical Summary")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        st.info("No numerical columns found.")
    else:
        st.dataframe(numeric_df.describe().T, use_container_width=True)

    st.subheader("Outlier Detection using IQR")
    excluded_for_analysis = classification_df.loc[
        classification_df["Role"].isin(["Identifier / Non-predictive", "Target"]), "Column"
    ].tolist()
    outlier_df = detect_outliers_iqr(df, excluded_cols=excluded_for_analysis)
    if outlier_df.empty:
        st.info("No numerical feature columns available for outlier detection.")
    else:
        st.dataframe(outlier_df, use_container_width=True)

    # -------------------- Visualizations --------------------

    st.header("5. Visual Analysis")

    numeric_cols = [
        col for col, role in zip(classification_df["Column"], classification_df["Role"])
        if role == "Numerical"
    ]

    if numeric_cols:
        selected_col = st.selectbox("Choose a numerical feature", numeric_cols)

        fig, ax = plt.subplots()
        ax.hist(df[selected_col].dropna(), bins=20)
        ax.set_title(f"Distribution of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numerical feature columns are available for visualization.")

    # -------------------- Recommendations --------------------

    st.header("6. Intelligent Preprocessing Recommendations")
    st.write(
        "Context-aware preprocessing recommendations combining smart column classification, "
        "missing-value analysis, outlier detection, category consistency analysis, and duplicate detection."
    )

    recommendations = make_recommendations(
        df,
        classification_df,
        outlier_df=outlier_df,
        consistency_df=consistency_df
    )

    if recommendations.empty:
        st.success("No major preprocessing issue was detected by the current rules.")
    else:
        st.dataframe(recommendations, use_container_width=True)

    st.caption(
        "The intelligent recommendation engine uses deterministic rules combining feature roles, missingness, "
        "IQR outlier metrics, category consistency, and duplicate checks."
    )

    # -------------------- AI-Powered Explanation --------------------

    st.header("7. AI-Powered Explanation")
    st.write(
        "The AI explanation layer uses Gemini to interpret each deterministic recommendation in plain language. "
        "A structured column summary (no raw data) is sent to Gemini for each flagged column."
    )

    # Read API key: prefer GEMINI_API_KEY env var / .env, allow manual entry as fallback
    _env_key = os.environ.get("GEMINI_API_KEY", "")
    _col_key, _col_model = st.columns([2, 1])
    with _col_key:
        _api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value="",
            placeholder="API Key detected from .env" if _env_key else "Paste your Gemini API key here (or set GEMINI_API_KEY in .env)",
            help=(
                "Your Gemini API key is used only for generating explanations in this session and is not stored. "
                "You can also set GEMINI_API_KEY in a .env file."
            ),
            key="gemini_api_key"
        )
        if _env_key and not _api_key_input:
            st.caption("🔑 *API key automatically detected from `.env`*")
    with _col_model:
        _selected_model = st.selectbox(
            "Gemini Model",
            options=[
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-flash-latest",
                "gemini-flash-lite-latest",
            ],
            index=0,
            help="Select primary Gemini model. Auto-fallback will be used if capacity or availability limits are hit.",
            key="gemini_model_choice"
        )
    _api_key = _api_key_input.strip() or _env_key

    if not _GEMINI_AVAILABLE:
        st.warning(
            "The `google-genai` package is not installed. "
            "Run `pip install google-genai` in your terminal and restart the app to enable AI explanations."
        )
    elif not _api_key:
        st.info(
            "Enter your Gemini API key above (or set the `GEMINI_API_KEY` environment variable) "
            "to enable AI-powered explanations. "
            "All existing deterministic analysis features work without an API key."
        )
    else:
        # Build lookup maps for outliers and consistency variants
        _outlier_map_ai = {}
        if not outlier_df.empty:
            for _, _row in outlier_df.iterrows():
                _outlier_map_ai[_row["Column"]] = int(_row.get("Outliers (IQR)", 0))

        _consistency_map_ai = {}
        if not consistency_df.empty:
            for _, _row in consistency_df.iterrows():
                _consistency_map_ai[_row["Column"]] = _row.get("Detected Variants", "")

        # Only explain non-trivial feature columns (skip whole-dataset rows)
        _roles_map = dict(zip(classification_df["Column"], classification_df["Role"]))
        _explainable_rows = recommendations[
            recommendations["Column"].isin(df.columns)
        ] if not recommendations.empty else pd.DataFrame()

        if _explainable_rows.empty:
            st.info("No columns with notable issues to explain.")
        else:
            if st.button("Generate AI Explanations", type="primary", key="ai_explain_btn"):
                st.session_state["ai_explanations"] = {}
                with st.spinner("Generating Gemini explanations..."):
                    for _, _rec in _explainable_rows.iterrows():
                        _col = _rec["Column"]
                        _role = _roles_map.get(_col, "Unknown")
                        _summary = build_column_summary(
                            _col, _role, df, _outlier_map_ai, _consistency_map_ai
                        )
                        _explanation = get_ai_explanation(
                            _api_key, _rec.to_dict(), _summary, preferred_model=_selected_model
                        )
                        st.session_state["ai_explanations"][_col] = _explanation

            if "ai_explanations" in st.session_state and st.session_state["ai_explanations"]:
                for _col, _explanation in st.session_state["ai_explanations"].items():
                    _rec_row = _explainable_rows[_explainable_rows["Column"] == _col]
                    if not _rec_row.empty:
                        _rec_dict = _rec_row.iloc[0].to_dict()
                        with st.expander(f"📋 {_col}  —  {_rec_dict.get('Detected Issues', '')}", expanded=True):
                            col_left, col_right = st.columns([1, 2])
                            with col_left:
                                st.markdown("**Recommended Action**")
                                st.write(_rec_dict.get("Recommended Action", ""))
                                st.markdown("**Risk/Note**")
                                st.write(_rec_dict.get("Risk/Note", ""))
                            with col_right:
                                st.markdown("**🤖 Gemini Explanation**")
                                if _explanation:
                                    st.info(_explanation)
                                else:
                                    st.warning("No explanation returned.")

    st.caption(
        "Gemini explanations are generated from a structured column summary only — "
        "no raw data rows are sent to the AI model."
    )

    # -------------------- User-Controlled Preprocessing Actions --------------------

    st.header("8. Preprocessing Actions")
    st.write(
        "Configure and explicitly select preprocessing operations to prepare your dataset for machine learning. "
        "All operations are performed strictly on a copy — your original uploaded dataset remains untouched."
    )

    # Identify columns by role
    _roles_dict = dict(zip(classification_df["Column"], classification_df["Role"]))
    _id_cols = [c for c, r in _roles_dict.items() if r == "Identifier / Non-predictive"]
    _numeric_feature_cols = [c for c, r in _roles_dict.items() if r == "Numerical"]
    _cat_feature_cols = [c for c, r in _roles_dict.items() if r == "Categorical"]

    # Missing values
    _missing_counts = df.isna().sum()
    _numeric_missing_cols = [c for c in _numeric_feature_cols if _missing_counts[c] > 0]
    _cat_missing_cols = [c for c in _cat_feature_cols if _missing_counts[c] > 0]

    # Duplicates
    _duplicate_rows_count = int(df.duplicated().sum())

    # Outliers
    _outlier_cols_with_counts = {}
    if not outlier_df.empty:
        for _, _row in outlier_df.iterrows():
            _cnt = int(_row.get("Outliers (IQR)", 0))
            if _cnt > 0:
                _outlier_cols_with_counts[_row["Column"]] = _cnt

    # 1. Identifier Columns Handling
    st.subheader("1. Identifier Columns Handling")
    if _id_cols:
        st.info(f"Detected identifier column(s): **{', '.join(_id_cols)}**")
        _handle_ids = st.checkbox(
            "Exclude identifier columns from ML feature transformations (Recommended)",
            value=True,
            help="Protects high-cardinality identifiers from being scaled or one-hot encoded, preventing feature explosion and overfitting.",
            key="chk_handle_ids"
        )
        _drop_ids = st.checkbox(
            "Drop identifier columns completely from cleaned dataset",
            value=False,
            help="If checked, identifier columns will be removed from the exported dataset.",
            key="chk_drop_ids"
        )
        st.caption(
            f"ℹ️ **Affected Columns:** {', '.join(_id_cols)} | "
            "**Explanation:** Identifiers label individual entities rather than predictive patterns. "
            "Excluding them from transformations prevents feature explosion and data leakage."
        )
    else:
        _handle_ids = True
        _drop_ids = False
        st.caption("No identifier columns detected in this dataset.")

    # 2. Duplicate Rows Removal
    st.subheader("2. Duplicate Rows Removal")
    if _duplicate_rows_count > 0:
        _remove_dups = st.checkbox(
            f"Remove duplicate rows ({_duplicate_rows_count} duplicate row(s) detected)",
            value=True,
            help="Removes identical repeated rows, keeping the first occurrence.",
            key="chk_remove_dups"
        )
        st.caption(
            f"ℹ️ **Affected:** Entire dataset ({_duplicate_rows_count} row(s)) | "
            "**Explanation:** Removing duplicates prevents biased statistical metrics and data leakage across splits."
        )
    else:
        _remove_dups = False
        st.caption("No duplicate rows detected in this dataset.")

    # 3. Inconsistent Categorical Values (Review & Standardization)
    st.subheader("3. Inconsistent Categorical Values Standardization")
    _confirmed_category_mappings = {}
    _unconfirmed_inconsistencies = []

    if not consistency_df.empty:
        st.write(
            "Review detected category variations below. "
            "Confirm each mapping explicitly before applying. Unconfirmed variants will **not** be modified."
        )
        _enable_standardize_cat = st.checkbox(
            "Standardize inconsistent categorical values",
            value=True,
            key="chk_enable_standardize_cat"
        )

        for _idx, _c_row in consistency_df.iterrows():
            _c_col = _c_row["Column"]
            _variants_str = _c_row["Detected Variants"]
            _suggested_val = _c_row["Suggested Standard Value"]
            _reason = _c_row["Reason"]

            _raw_variants = [v.strip() for v in str(_variants_str).split(",") if v.strip()]
            _non_standard_variants = [v for v in _raw_variants if v != _suggested_val]

            with st.expander(f"Review Mappings for '{_c_col}' (Suggested Standard: '{_suggested_val}')", expanded=True):
                st.markdown(f"**Detected Variants:** `{_variants_str}`")
                st.markdown(f"**Diagnostic Details:** {_reason}")

                _col_mappings = {}
                for _v in _non_standard_variants:
                    _is_formatting_only = (
                        " ".join(_v.strip().casefold().split()) == " ".join(_suggested_val.strip().casefold().split())
                    )
                    _c_check, _c_val = st.columns([2, 1])
                    with _c_check:
                        _default_confirm = bool(_is_formatting_only)
                        _diff_label = " (Formatting difference)" if _is_formatting_only else " ⚠️ (Spelling/Spacing variation - confirm mapping)"
                        _confirm = st.checkbox(
                            f"Standardize '{_v}' ➔ '{_suggested_val}'{_diff_label}",
                            value=_default_confirm,
                            key=f"confirm_map_{_c_col}_{_v}"
                        )
                    with _c_val:
                        _target_val = st.text_input(
                            "Target Canonical Value",
                            value=_suggested_val,
                            key=f"target_val_{_c_col}_{_v}",
                            label_visibility="collapsed"
                        )

                    if _confirm and _enable_standardize_cat:
                        _col_mappings[_v] = _target_val
                    else:
                        _unconfirmed_inconsistencies.append({
                            "col": _c_col,
                            "variants": _v,
                            "standard": _suggested_val
                        })

                if _col_mappings:
                    _confirmed_category_mappings[_c_col] = _col_mappings

        st.caption(
            f"ℹ️ **Affected Columns:** {', '.join(consistency_df['Column'].unique())} | "
            "**Explanation:** Standardizing formatting and spelling variations unifies synonymous category entries, "
            "preventing fragmented dummy columns during encoding."
        )
    else:
        _enable_standardize_cat = False
        st.caption("No categorical consistency issues detected in this dataset.")

    # 4. Missing Value Imputation
    st.subheader("4. Missing Value Imputation")
    _col_imp_num, _col_imp_cat = st.columns(2)

    with _col_imp_num:
        st.markdown("#### Numerical Imputation")
        if _numeric_missing_cols:
            _impute_num = st.checkbox(
                "Impute numerical missing values",
                value=True,
                key="chk_impute_num"
            )
            _num_strategy = st.radio(
                "Numerical Imputation Method",
                ["Median (Recommended - robust to outliers)", "Mean"],
                index=0,
                key="radio_num_strategy"
            )
            _selected_num_cols = st.multiselect(
                "Numerical columns to impute",
                options=_numeric_missing_cols,
                default=_numeric_missing_cols,
                key="ms_num_cols"
            )
            st.caption(
                f"ℹ️ **Affected Columns:** {', '.join(_selected_num_cols) if _selected_num_cols else 'None'} | "
                "**Explanation:** Fills numerical gaps with median (robust against outliers) or mean."
            )
        else:
            _impute_num = False
            _num_strategy = "Median"
            _selected_num_cols = []
            st.caption("No numerical feature columns contain missing values.")

    with _col_imp_cat:
        st.markdown("#### Categorical Imputation")
        if _cat_missing_cols:
            _impute_cat = st.checkbox(
                "Impute categorical missing values (Mode)",
                value=True,
                key="chk_impute_cat"
            )
            _selected_cat_cols = st.multiselect(
                "Categorical columns to impute",
                options=_cat_missing_cols,
                default=_cat_missing_cols,
                key="ms_cat_cols"
            )
            st.caption(
                f"ℹ️ **Affected Columns:** {', '.join(_selected_cat_cols) if _selected_cat_cols else 'None'} | "
                "**Explanation:** Replaces missing categories with the most frequent value (mode) of that column."
            )
        else:
            _impute_cat = False
            _selected_cat_cols = []
            st.caption("No categorical feature columns contain missing values.")

    # 5. Numerical Outliers Review
    st.subheader("5. Numerical Outliers Review")
    if _outlier_cols_with_counts:
        st.write(
            "Outliers were detected outside the 1.5 × IQR boundary. "
            "Review detected outliers below. Outliers are **never** automatically removed."
        )
        for _ocol, _ocnt in _outlier_cols_with_counts.items():
            _s = df[_ocol].dropna()
            _q1 = float(_s.quantile(0.25))
            _q3 = float(_s.quantile(0.75))
            _iqr = _q3 - _q1
            _low = _q1 - 1.5 * _iqr
            _up = _q3 + 1.5 * _iqr
            _ext = _s[(_s < _low) | (_s > _up)].tolist()
            st.write(f"• **{_ocol}**: {_ocnt} outlier value(s) outside [{_low:.2f}, {_up:.2f}]. Sample values: `{_ext[:5]}`")

        _outlier_action = st.radio(
            "Outlier Strategy",
            [
                "Keep outliers as-is (Recommended to preserve natural extreme variation)",
                "Cap outliers at IQR bounds (Winsorization: clamp values outside 1.5×IQR)",
                "Remove rows containing outliers (Caution: drops records with extreme values)"
            ],
            index=0,
            key="radio_outlier_action"
        )
        _selected_outlier_cols = st.multiselect(
            "Columns to apply outlier action to",
            options=list(_outlier_cols_with_counts.keys()),
            default=list(_outlier_cols_with_counts.keys()),
            key="ms_outlier_cols"
        )
        st.caption(
            f"ℹ️ **Affected Columns:** {', '.join(_selected_outlier_cols) if _selected_outlier_cols else 'None'} | "
            "**Explanation:** Outliers can represent measurement noise or critical real-world phenomena. "
            "Preserving them avoids loss of valid data points, while winsorization caps extreme skew."
        )
    else:
        _outlier_action = "Keep outliers as-is (Recommended to preserve natural extreme variation)"
        _selected_outlier_cols = []
        st.caption("No IQR outliers detected in numerical feature columns.")

    # 6. Numerical Feature Scaling
    st.subheader("6. Numerical Feature Scaling")
    if _numeric_feature_cols:
        _scale_num = st.checkbox(
            "Scale numerical features",
            value=False,
            key="chk_scale_num",
            help="Optional transformation to standardize or normalize feature scales."
        )
        _col_sc1, _col_sc2 = st.columns([1, 2])
        with _col_sc1:
            _scaler_type = st.radio(
                "Scaling Technique",
                ["StandardScaler (zero mean, unit variance)", "MinMaxScaler (scale to range [0, 1])"],
                key="radio_scaler_type"
            )
        with _col_sc2:
            _selected_scale_cols = st.multiselect(
                "Numerical feature columns to scale",
                options=_numeric_feature_cols,
                default=_numeric_feature_cols,
                key="ms_scale_cols"
            )
        st.caption(
            f"ℹ️ **Affected Columns:** {', '.join(_selected_scale_cols) if _selected_scale_cols else 'None'} | "
            "**Explanation:** Rescaling features ensures algorithms sensitive to magnitude treat all features proportionally."
        )
    else:
        _scale_num = False
        _scaler_type = "StandardScaler"
        _selected_scale_cols = []
        st.caption("No numerical feature columns available for scaling.")

    # 7. Categorical Feature Encoding
    st.subheader("7. Categorical Feature Encoding")
    if _cat_feature_cols:
        _encode_cat = st.checkbox(
            "Encode categorical features (One-Hot Encoding)",
            value=False,
            key="chk_encode_cat",
            help="Converts discrete categories into binary 0/1 indicator columns for machine learning algorithms."
        )
        _col_enc1, _col_enc2 = st.columns([1, 2])
        with _col_enc1:
            _encode_drop_first = st.checkbox(
                "Drop first dummy category (prevents multicollinearity)",
                value=False,
                key="chk_drop_first"
            )
        with _col_enc2:
            _selected_encode_cols = st.multiselect(
                "Categorical feature columns to encode",
                options=_cat_feature_cols,
                default=_cat_feature_cols,
                key="ms_encode_cols"
            )
        st.caption(
            f"ℹ️ **Affected Columns:** {', '.join(_selected_encode_cols) if _selected_encode_cols else 'None'} | "
            "**Explanation:** One-hot encoding produces binary indicator columns for ML models. "
            "Identifiers and target columns are strictly excluded."
        )
    else:
        _encode_cat = False
        _encode_drop_first = False
        _selected_encode_cols = []
        st.caption("No categorical feature columns available for encoding.")

    # -------------------- Apply Button & Results --------------------

    st.divider()

    if st.button("Apply Selected Preprocessing", type="primary", key="apply_preprocessing_btn"):
        _actions_config = {
            "handle_identifiers": _handle_ids,
            "drop_identifiers": _drop_ids,
            "remove_duplicates": _remove_dups,
            "standardize_categories": _enable_standardize_cat,
            "category_mappings": _confirmed_category_mappings,
            "unconfirmed_inconsistencies": _unconfirmed_inconsistencies,
            "impute_numeric": _impute_num,
            "numeric_strategy": "Mean" if "Mean" in _num_strategy else "Median",
            "numeric_cols": _selected_num_cols,
            "impute_categorical": _impute_cat,
            "categorical_cols": _selected_cat_cols,
            "outlier_action": _outlier_action,
            "outlier_cols": _selected_outlier_cols,
            "scale_numeric": _scale_num,
            "scaler_type": "MinMaxScaler" if "MinMax" in _scaler_type else "StandardScaler",
            "scale_cols": _selected_scale_cols,
            "encode_categories": _encode_cat,
            "encode_drop_first": _encode_drop_first,
            "encode_cols": _selected_encode_cols,
        }

        _cleaned_df, _applied_log, _unresolved_issues, _errors_log = apply_user_controlled_preprocessing(
            df,
            classification_df,
            _actions_config
        )

        st.session_state["cleaned_dataset_result"] = {
            "cleaned_df": _cleaned_df,
            "applied_log": _applied_log,
            "unresolved_issues": _unresolved_issues,
            "errors_log": _errors_log,
        }
        st.success("Selected preprocessing completed successfully!")

    if "cleaned_dataset_result" in st.session_state:
        _res = st.session_state["cleaned_dataset_result"]
        _cleaned_df = _res["cleaned_df"]
        _applied_log = _res["applied_log"]
        _unresolved_issues = _res["unresolved_issues"]
        _errors_log = _res["errors_log"]

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(_cleaned_df.head(10), use_container_width=True)

        st.subheader("Before vs. After Comparison")

        _before_rows = df.shape[0]
        _after_rows = _cleaned_df.shape[0]
        _before_cols = df.shape[1]
        _after_cols = _cleaned_df.shape[1]
        _before_missing = int(df.isna().sum().sum())
        _after_missing = int(_cleaned_df.isna().sum().sum())
        _before_dups = int(df.duplicated().sum())
        _after_dups = int(_cleaned_df.duplicated().sum())

        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Rows", f"{_before_rows} ➔ {_after_rows}", delta=f"{_after_rows - _before_rows}")
        _mc2.metric("Columns", f"{_before_cols} ➔ {_after_cols}", delta=f"{_after_cols - _before_cols}")
        _mc3.metric("Missing Cells", f"{_before_missing} ➔ {_after_missing}", delta=f"{_after_missing - _before_missing}")
        _mc4.metric("Duplicate Rows", f"{_before_dups} ➔ {_after_dups}", delta=f"{_after_dups - _before_dups}")

        _comparison_df = pd.DataFrame({
            "Metric": ["Row Count", "Column Count", "Missing Cells", "Duplicate Rows"],
            "Original Dataset": [_before_rows, _before_cols, _before_missing, _before_dups],
            "Cleaned Dataset": [_after_rows, _after_cols, _after_missing, _after_dups],
            "Change": [
                _after_rows - _before_rows,
                _after_cols - _before_cols,
                _after_missing - _before_missing,
                _after_dups - _before_dups,
            ]
        })
        st.dataframe(_comparison_df, use_container_width=True)

        st.subheader("Selected Transformations & Actions Applied")
        if _applied_log:
            st.dataframe(pd.DataFrame(_applied_log), use_container_width=True)
        else:
            st.info("No preprocessing transformations were selected or applied.")

        st.subheader("Unresolved Issues & Observations")
        if _unresolved_issues:
            st.warning("The following issues or considerations remain unaddressed per your configuration:")
            for _u in _unresolved_issues:
                st.markdown(f"- ⚠️ {_u}")
        else:
            st.info("✅ All detected data quality, formatting, and preprocessing requirements have been resolved.")

        if _errors_log:
            st.error("Warnings / Errors encountered during processing:")
            for _e in _errors_log:
                st.markdown(f"- ❌ {_e}")

        st.subheader("Download Cleaned Dataset & Report")
        _col_dl1, _col_dl2 = st.columns(2)

        with _col_dl1:
            _csv_bytes = _cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Cleaned CSV",
                data=_csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        with _col_dl2:
            _report_text = generate_preprocessing_report(
                df,
                _cleaned_df,
                _applied_log,
                _unresolved_issues,
                _errors_log
            )
            st.download_button(
                "📄 Download Preprocessing Report (Markdown)",
                data=_report_text.encode("utf-8"),
                file_name="preprocessing_report.md",
                mime="text/markdown",
                use_container_width=True
            )

if __name__ == "__main__" or (hasattr(st, "runtime") and st.runtime.exists()):
    main()
