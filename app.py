from flask import Flask, json, render_template, request, jsonify
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os
import pickle

app = Flask(__name__)

@app.route('/train-multinomial', methods=['POST'])
def train_multinomial():
    # Parse JSON data from the request
    k_values = json.loads(request.form['k_values'])

    file = request.files['file']
    file_path = os.path.join('./assets/dataset', 'multinomial.xlsx')
    file.save(file_path)

    # load dataset
    file_path = os.path.join('./assets/dataset', 'multinomial.xlsx')
    df = pd.read_excel(file_path)

    # Mapping 
    ekspedisi_map = dict(enumerate(df['Ekspedisi'].astype('category').cat.categories))
    item_map = dict(enumerate(df['Item'].astype('category').cat.categories))

    # Data Preprocessing
    df['Ekspedisi'] = df['Ekspedisi'].astype('category').cat.codes
    df['Item'] = df['Item'].astype('category').cat.codes

    # # Mapping Ekspedisi code to labels (optional: useful for interpreting results)
    ekspedisi_mapping = {code: category for code, category in enumerate(df['Ekspedisi'].astype('category').cat.categories)}

    # Features (X) dan target (y)
    X = df[['Item', 'Ekspedisi', 'Jumlah Barang yang Di Pick Up', 'Jumlah Barang yang Rusak']]
    y = df['keterangan'].apply(lambda x: 1 if x == 'rusak' else 0)  # rusak = 1, tidak rusak = 0

    # Check the distribution of the target labels
    print("Distribusi label pada data:")
    print(y.value_counts())
    print("\nLabel '1' mewakili barang rusak, dan '0' mewakili barang tidak rusak.")

    # Model Naive Bayes Multinomial
    model = MultinomialNB()

    # Initialize a dictionary to store accuracy for each K-Fold
    accuracy_dict = {}
    predictions_dict = {}

    # List of K values for KFold Cross Validation
    # k_values = [3, 4, 6]

    # Iterate over each K and perform K-Fold Cross Validation
    for k in k_values:
        print(f"\nK-Fold: {k}")

        # K-Fold Cross Validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # Cross-validation prediction
        y_pred = cross_val_predict(model, X, y, cv=kf)

        # Save Model
        # 4. Fit the model on the entire training set
        model.fit(X, y_pred)

        # 5. Export the model using pickle
        with open(f"./assets/model/multinomial_k{k}.pkl", "wb") as f:
            pickle.dump(model, f)

        # Store predictions for later saving
        predictions_dict[f"K={k}"] = y_pred

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        print(f"Confusion Matrix (K={k}):\n", cm)

        # Classification report
        cr = classification_report(y, y_pred, target_names=['Tidak Rusak', 'Rusak'])
        print(f"Classification Report (K={k}):\n", cr)

        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)
        accuracy_dict[k] = accuracy
        print(f"Akurasi untuk K={k}: {accuracy:.4f}")

        # Save predictions for each K-Fold into a separate Excel file
        predictions_df = pd.DataFrame({
            'Item': df['Item'].map(lambda x: item_map[x]),
            'Ekspedisi': df['Ekspedisi'].map(lambda x: ekspedisi_map[x]),
            'Jumlah Barang di Pick Up': df['Jumlah Barang yang Di Pick Up'],
            'Jumlah Barang Rusak': df['Jumlah Barang yang Rusak'],
            'Keterangan Asli': df['keterangan'],
            f'Prediksi K-Fold {k}': y_pred
        })

        # Map numerical predictions back to labels (1 = rusak, 0 = tidak rusak)
        predictions_df[f'Prediksi K-Fold {k}'] = predictions_df[f'Prediksi K-Fold {k}'].map({1: 'rusak', 0: 'tidak rusak'})

        # Save the predictions to an Excel file
        output_file = f'./assets/train/prediksi_multinomial_kerusakan_k{k}.xlsx'
        predictions_df.to_excel(output_file, index=False)

        # Analysis: Find expedition with lowest predicted damage
        # Group by Ekspedisi and calculate the percentage of items predicted to be damaged
        ekspedisi_damage = predictions_df.groupby('Ekspedisi')[f'Prediksi K-Fold {k}'].apply(lambda x: (x == 'rusak').mean())
        ekspedisi_damage = ekspedisi_damage.rename(index=ekspedisi_mapping)

        print(f"\nTingkat kerusakan per ekspedisi untuk K={k}:")
        print(ekspedisi_damage)

        ekspedisi_terendah = ekspedisi_damage.idxmin()
        persentase_terendah = ekspedisi_damage.min()

        print(f"Ekspedisi dengan tingkat kerusakan terendah untuk K={k}: {ekspedisi_terendah} ({persentase_terendah:.2%})")

    # Find the K-Fold with the highest accuracy
    best_k = max(accuracy_dict, key=accuracy_dict.get)
    best_accuracy = accuracy_dict[best_k]
    print(f"\nK-Fold dengan akurasi paling tinggi: {best_k}, dengan akurasi: {best_accuracy:.4f}")

    # Write conclusion to a text file
    conclusion = f"K-Fold dengan akurasi paling tinggi adalah K={best_k}, dengan akurasi: {best_accuracy:.4f}. " \
                "Nilai K yang lebih besar memberikan hasil yang lebih akurat karena lebih banyak pembagian data " \
                "melalui fold yang membantu model untuk generalisasi lebih baik, namun masih mempertahankan jumlah data pelatihan yang cukup."
    
    # read excel result
    excel = pd.read_excel(f'./assets/train/prediksi_multinomial_kerusakan_k{best_k}.xlsx')
    excelData = excel.to_dict(orient='records')

    response = {
         "conclusion": conclusion,
         "accuracy": best_accuracy,
         "best_model": best_k,
         "result": excelData,
         "ekspedisi_map": ekspedisi_map,
         "item_map": item_map,
    }

    return jsonify(response)

@app.route('/predict-multinomial')
def predict_multinomial():
    item = request.args.get('item')
    ekspedisi = request.args.get('ekspedisi')
    pickup = request.args.get('pickup')
    rusak = request.args.get('rusak')
    k = request.args.get('k')

    # 1. Load the saved model
    with open(f"./assets/model/multinomial_k{k}.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. Prepare input data for prediction
    # Example input: Replace with your actual test data
    new_data = np.array([[int(item), int(ekspedisi), int(pickup), int(rusak)]])

    # 3. Predict the label
    predictions = model.predict(new_data)

    response = {
         "prediction": predictions[0].item()
    }

    return jsonify(response)

@app.route('/multinomial')
def multinomial():
     return render_template('multinomial.html')

if __name__ == '__main__':
	app.run(debug=True)