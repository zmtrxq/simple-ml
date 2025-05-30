from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Plotting Helper Functions (Identical to previous full response) ---
def _create_plot_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def generate_correlation_matrix_plot(df_numeric):
    if df_numeric.empty or df_numeric.shape[1] < 2: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 8})
        ax.set_title('Correlation Matrix of Features')
        plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
        return _create_plot_base64(fig)
    except Exception as e:
        app.logger.error(f"Correlation matrix generation error: {e}"); plt.close(fig); return None

def generate_feature_importance_plot(model, feature_names, model_name):
    if not hasattr(model, 'feature_importances_'): return None
    importances = model.feature_importances_
    if len(feature_names) != len(importances):
        app.logger.warning(f"FI plot: Mismatch names ({len(feature_names)}) vs importances ({len(importances)}) for {model_name}.")
        if len(importances) > 0: feature_names = [f"Feature {i}" for i in range(len(importances))]
        else: return None
    feature_importance_tuples = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_k = min(len(feature_importance_tuples), 15)
    top_features = feature_importance_tuples[:top_k]
    plot_labels = [f[0] for f in top_features][::-1]
    plot_values = [f[1] for f in top_features][::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    ax.set_title(f"Feature Importances ({model_name} - Top {top_k})")
    ax.barh(plot_labels, plot_values, align="center", color="skyblue")
    ax.set_xlabel("Importance")
    return _create_plot_base64(fig)

def generate_prediction_plot_regression(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', s=50, color="crimson")
    min_val, max_val = (min(y_true.min(),y_pred.min()), max(y_true.max(),y_pred.max())) if len(y_true)>0 and len(y_pred)>0 else (0,1)
    if min_val == max_val: max_val = min_val + 1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_xlabel('Actual Values'); ax.set_ylabel('Predicted Values')
    ax.set_title(f'Actual vs. Predicted ({model_name})'); ax.grid(True, linestyle=':', alpha=0.7)
    return _create_plot_base64(fig)

# --- Core Logic ---
def get_custom_default_preprocessor(X_df):
    # (Identical to previous full response)
    num_cols = X_df.select_dtypes(exclude=['object']).columns.tolist()
    cat_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    transformers_list = []
    if num_cols: transformers_list.append(('num', numerical_transformer, num_cols))
    if cat_cols: transformers_list.append(('cat', categorical_transformer, cat_cols))
    if not transformers_list:
        app.logger.warning("No numerical or categorical columns identified for preprocessor.")
        return ColumnTransformer([('passthrough_all', 'passthrough', list(range(X_df.shape[1])))], remainder='drop')
    return ColumnTransformer(transformers_list, remainder='passthrough')

def get_feature_names_after_preprocessing(preprocessor, num_transformed_cols):
    # (Identical to previous full response)
    try: return preprocessor.get_feature_names_out()
    except AttributeError: app.logger.warning("preprocessor.get_feature_names_out() failed. Using generic names.")
    except Exception as e: app.logger.error(f"Error in get_feature_names_out: {e}. Using generic names.")
    return [f"feature_{i}" for i in range(num_transformed_cols)]

@app.route('/')
def index_route():
    # (Identical to previous full response)
    return render_template('index.html')

@app.route('/get_columns', methods=['POST'])
def get_columns_route():
    # (Identical to previous full response)
    if 'train_file' not in request.files: return jsonify({'error': 'No training file.'}), 400
    train_file = request.files['train_file']
    if not train_file.filename: return jsonify({'error': 'No selected file.'}), 400
    try:
        df = pd.read_csv(train_file)
        df_head_html = df.head().to_html(classes=['dataframe-head', 'neon-table'], border=0, index=False).replace('<th>', '<th style="text-align: left;">')
        return jsonify({'columns': df.columns.tolist(), 'df_head_html': df_head_html})
    except Exception as e:
        app.logger.error(f"Error in /get_columns: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Could not read CSV: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_data_route():
    try:
        train_file = request.files.get('train_file'); test_file = request.files.get('test_file')
        target_column_name_from_user = request.form.get('target_column') # This is the original target name
        problem_type = request.form.get('problem_type')
        columns_to_drop = json.loads(request.form.get('columns_to_drop', '[]'))

        if not train_file or not test_file: return jsonify({'error': 'Train/Test file missing.'}), 400
        if not target_column_name_from_user: return jsonify({'error': 'Target column missing.'}), 400

        train_df_full = pd.read_csv(train_file); test_df_full = pd.read_csv(test_file)

        if target_column_name_from_user not in train_df_full.columns: return jsonify({'error': 'Target not in train data.'}), 400
        if target_column_name_from_user in columns_to_drop: return jsonify({'error': 'Target cannot be dropped.'}), 400

        train_df = train_df_full.drop(columns=[c for c in columns_to_drop if c in train_df_full.columns], errors='ignore')
        X_train_raw = train_df.drop(columns=[target_column_name_from_user], errors='ignore')
        y_train_raw = train_df[target_column_name_from_user]

        test_df_cols_to_drop = [c for c in columns_to_drop if c in test_df_full.columns]
        # If target exists in test_df_full, it should also be dropped from X_test_raw features
        if target_column_name_from_user in test_df_full.columns and target_column_name_from_user not in test_df_cols_to_drop:
            test_df_cols_to_drop.append(target_column_name_from_user)
        X_test_raw = test_df_full.drop(columns=test_df_cols_to_drop, errors='ignore')
        
        # Align test columns to train (after drops)
        original_train_feature_cols = X_train_raw.columns.tolist() # Columns used for training
        for col in original_train_feature_cols:
            if col not in X_test_raw.columns: X_test_raw[col] = 0 # Add missing feature cols to test
        # Ensure test set only has features that were in training (and in same order)
        X_test_raw = X_test_raw[original_train_feature_cols]


        correlation_plot_b64 = generate_correlation_matrix_plot(X_train_raw.select_dtypes(include=np.number))

        label_encoder = None; y_train = y_train_raw.copy()
        if problem_type == 'classification':
            if y_train_raw.dtype == 'object' or pd.api.types.is_categorical_dtype(y_train_raw):
                label_encoder = LabelEncoder(); y_train = label_encoder.fit_transform(y_train_raw)
            else: y_train = y_train_raw.astype(int)
        
        preprocessor = get_custom_default_preprocessor(X_train_raw)
        X_train_processed = preprocessor.fit_transform(X_train_raw)
        X_test_processed = preprocessor.transform(X_test_raw) 
        
        processed_feature_names = get_feature_names_after_preprocessing(preprocessor, X_train_processed.shape[1])
        if processed_feature_names is None or len(processed_feature_names) != X_train_processed.shape[1]:
             processed_feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

        models_def = { # (Identical model definitions)
            'classification': {'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42, max_iter=200),
                               'Random Forest Classifier': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                               'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=50, random_state=42)},
            'regression': {'Linear Regression': LinearRegression(n_jobs=-1),
                             'Random Forest Regressor': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                             'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=50, random_state=42)}
        }
        cv_scoring_metrics_map = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 
                                  'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro'} \
                                 if problem_type == 'classification' else \
                                 {'r2': 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'}

        results_output = []; summary_metrics_data = []
        for model_name, model_instance in models_def[problem_type].items():
            current_model_pipeline = Pipeline(steps=[('model', model_instance)])
            cv_results = cross_validate(current_model_pipeline, X_train_processed, y_train, 
                                        cv=3, scoring=cv_scoring_metrics_map, n_jobs=-1, return_train_score=False)
            model_detailed_cv_metrics = {}; summary_row = {'Model': model_name}
            for metric_key, scores_array in cv_results.items():
                if metric_key.startswith('test_'):
                    mean_score = scores_array.mean()
                    display_name_detailed = metric_key.replace('test_', '').replace('_macro', ' (macro)').replace('neg_mean_squared_error', 'Neg MSE').capitalize()
                    if display_name_detailed == 'Neg mse': model_detailed_cv_metrics['CV RMSE'] = round(np.sqrt(-mean_score), 4)
                    else: model_detailed_cv_metrics[f"CV {display_name_detailed}"] = round(mean_score, 4)
                    if problem_type == 'classification':
                        if metric_key == 'test_accuracy': summary_row['Accuracy'] = round(mean_score, 4)
                        elif metric_key == 'test_f1_macro': summary_row['F1 (macro)'] = round(mean_score, 4)
                        elif metric_key == 'test_precision_macro': summary_row['Precision (macro)'] = round(mean_score, 4)
                        elif metric_key == 'test_recall_macro': summary_row['Recall (macro)'] = round(mean_score, 4)
                    elif problem_type == 'regression':
                        if metric_key == 'test_r2': summary_row['R2 Score'] = round(mean_score, 4)
                        elif metric_key == 'test_neg_mean_squared_error': summary_row['RMSE'] = round(np.sqrt(-mean_score), 4)
            if summary_row and len(summary_row) > 1 : summary_metrics_data.append(summary_row) # Ensure row has metrics
            
            current_model_pipeline.fit(X_train_processed, y_train)
            fi_plot = generate_feature_importance_plot(current_model_pipeline.named_steps['model'], processed_feature_names, model_name)
            pred_plot = None
            if problem_type == 'regression':
                pred_plot = generate_prediction_plot_regression(y_train_raw, current_model_pipeline.predict(X_train_processed), model_name)
            
            # --- MODIFICATION: Get and store final test predictions ---
            test_preds_numeric = current_model_pipeline.predict(X_test_processed)
            test_preds_final = test_preds_numeric # Default for regression or if no label encoder
            if problem_type == 'classification' and label_encoder:
                try:
                    test_preds_final = label_encoder.inverse_transform(test_preds_numeric.astype(int))
                except Exception as le_err:
                    app.logger.warning(f"Could not inverse transform test predictions for {model_name}: {le_err}. Using numeric preds.")
                    test_preds_final = test_preds_numeric # Fallback to numeric if inverse transform fails
            
            results_output.append({'model': model_name, 'metrics': model_detailed_cv_metrics,
                                   'feature_importance_plot': fi_plot, 'prediction_plot': pred_plot,
                                   'test_set_predictions': test_preds_final.tolist() # Add this line
                                  })
        
        if summary_metrics_data: # Sort summary table
            sort_key = 'Accuracy' if problem_type == 'classification' else 'R2 Score'
            if summary_metrics_data and sort_key in summary_metrics_data[0]:
                 summary_metrics_data = sorted(summary_metrics_data, key=lambda x: x.get(sort_key, 0), reverse=True)
            elif problem_type == 'regression' and summary_metrics_data and 'RMSE' in summary_metrics_data[0]:
                 summary_metrics_data = sorted(summary_metrics_data, key=lambda x: x.get('RMSE', float('inf')), reverse=False)

        return jsonify({'results': results_output, 'correlation_plot': correlation_plot_b64,
                        'summary_metrics': summary_metrics_data, 'problem_type': problem_type,
                        'target_column_name': target_column_name_from_user # Pass original target name for CSV header
                       })
    except Exception as e:
        app.logger.error(f"Error in /process: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)