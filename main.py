import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import glob
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from tensorflow.keras.utils import plot_model

tf.random.set_seed(42)
np.random.seed(42)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

class DiabetesMultiModalDataLoader:
    """Caricatore dati per dataset HUPA-UCM"""
    
    def __init__(self, data_path="dataset/preprocessed"):
        self.data_path = data_path
        self.scaler_cgm = StandardScaler()
        self.scaler_fitbit = StandardScaler()
        self.scaler_insulin = StandardScaler()
        self.sequence_length = 96  # 24 ore * 4 letture/ora (ogni 15 min)
        self.prediction_horizon = 16  # 4 ore future

    def train_val_test_split(self, data, random_state=42):
        """
        Divide i dati in train/validation/test
        """

        print(f"\nSplit dati per paziente...")
        
        # Ottieni pazienti unici
        unique_patients = np.unique(data['patient_ids'])

        train_data, temp_data = train_test_split(
            unique_patients, test_size=0.4, random_state=random_state, stratify=None
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=random_state
        )

        print(f"\nSplit pazienti:")
        print(f"  Training: {len(train_data)} pazienti")
        print(f"  Validation: {len(val_data)} pazienti")
        print(f"  Test: {len(test_data)} pazienti")
        
        return train_data, val_data, test_data
    
    def prepare_data(self, data, train_data, val_data, test_data):
        """Prepara i dati per il training"""
        # Maschere per split
        train_mask = np.isin(data['patient_ids'], train_data)
        val_mask = np.isin(data['patient_ids'], val_data)
        test_mask = np.isin(data['patient_ids'], test_data)
        
        # Dataset splits
        X_train = {
            'cgm': data['cgm'][train_mask],
            'fitbit': data['fitbit'][train_mask],
            'insulin': data['insulin'][train_mask]
        }
        X_val = {
            'cgm': data['cgm'][val_mask],
            'fitbit': data['fitbit'][val_mask],
            'insulin': data['insulin'][val_mask]
        }
        X_test = {
            'cgm': data['cgm'][test_mask],
            'fitbit': data['fitbit'][test_mask],
            'insulin': data['insulin'][test_mask]
        }
        
        y_train = data['target'][train_mask]
        y_val = data['target'][val_mask]
        y_test = data['target'][test_mask]
        
        print(f"\nDimensioni dataset:")
        print(f"- Training set: {len(y_train)} campioni")
        print(f"- Validation set: {len(y_val)} campioni")
        print(f"- Test set: {len(y_test)} campioni")
        print(f"- CGM sequence shape: {X_train['cgm'].shape}")
        print(f"- Fitbit sequence shape: {X_train['fitbit'].shape}")
        print(f"- Insulin features shape: {X_train['insulin'].shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test
        
    def load_patient_data(self, patient_file):
        """Carica dati di un singolo paziente"""
        try:
            # Lettura CSV, separatore: punto e virgola
            df = pd.read_csv(patient_file, sep=';')
            
            # Conversione timestamp in datetime
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            
            # Gestione valori mancanti
            df['glucose'] = df['glucose'].interpolate(method='linear')
            df['calories'] = df['calories'].fillna(df['calories'].mean())
            df['heart_rate'] = df['heart_rate'].fillna(df['heart_rate'].mean())
            df['steps'] = df['steps'].fillna(0)
            df['basal_rate'] = df['basal_rate'].fillna(df['basal_rate'].mean())
            df['bolus_volume_delivered'] = df['bolus_volume_delivered'].fillna(0)
            df['carb_input'] = df['carb_input'].fillna(0)
            
            # Rimozione outlier estremi
            df['glucose'] = np.clip(df['glucose'], 40, 500)
            df['heart_rate'] = np.clip(df['heart_rate'], 40, 200)
            
            return df
            
        except Exception as e:
            print(f"Errore nel caricamento di {patient_file}: {e}")
            return None
    
    def create_sequences(self, df, patient_id):
        """Crea sequenze temporali per training"""
        sequences = []
        
        max_sequences = len(df) - self.sequence_length - self.prediction_horizon
        
        for i in range(0, max_sequences, self.sequence_length // 4):  # Overlap del 75%
            
            # Sequenza input (24 ore)
            start_idx = i
            end_idx = i + self.sequence_length
            
            # Verifica che abbiamo abbastanza dati
            if end_idx + self.prediction_horizon >= len(df):
                break
                
            # Estrazione features
            sequence_data = df.iloc[start_idx:end_idx]
            future_data = df.iloc[end_idx:end_idx + self.prediction_horizon]
            
            # CGM sequence
            cgm_seq = sequence_data['glucose'].values
            
            # Fitbit sequence  
            fitbit_seq = np.column_stack([
                sequence_data['steps'].values,
                sequence_data['calories'].values,
                sequence_data['heart_rate'].values
            ])
            
            # Insulin/carb features (aggregati su finestra temporale)
            total_bolus = sequence_data['bolus_volume_delivered'].sum()
            avg_basal = sequence_data['basal_rate'].mean()
            total_carbs = sequence_data['carb_input'].sum()
            
            # Target: iperglicemia nelle prossime 4 ore
            future_glucose = future_data['glucose'].values
            max_future_glucose = np.max(future_glucose) if len(future_glucose) > 0 else 0
            hyperglycemia_target = int(max_future_glucose > 180)
            
            if (len(cgm_seq) == self.sequence_length and 
                not np.any(np.isnan(cgm_seq)) and
                len(future_glucose) > 0):
                
                sequences.append({
                    'patient_id': patient_id,
                    'cgm_sequence': cgm_seq,
                    'fitbit_sequence': fitbit_seq,
                    'total_bolus': total_bolus,
                    'avg_basal': avg_basal,
                    'total_carbs': total_carbs,
                    'target_hyperglycemia': hyperglycemia_target,
                    'max_future_glucose': max_future_glucose
                })
        
        return sequences
    
    def analyze_data_distribution(self, sequences, patient_ids, is_raw_data=True):
        """Analizza la distribuzione dei dati"""
        
        # Determina le label in base al tipo di dati
        if is_raw_data:
            title_suffix = "(Dati Grezzi)"
            filename = "plots/data_distribution_raw.png"
        else:
            title_suffix = "(Dati Preprocessati)" 
            filename = "plots/data_distribution_preprocessed.png"
        
        targets = [int(seq['max_future_glucose'] > 180) for seq in sequences]
        target_dist = np.bincount(targets)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Distribuzione target
        axes[0,0].bar(['No Iperglicemia', 'Iperglicemia'], target_dist, color=['skyblue', 'salmon'])
        axes[0,0].set_title(f'Distribuzione Target {title_suffix}')
        axes[0,0].set_ylabel('Numero Campioni')
        
        # Calcola tasso iperglicemia per paziente
        patient_stats = []
        for patient_id in np.unique(patient_ids):
            patient_mask = np.array(patient_ids) == patient_id
            patient_samples = np.sum(patient_mask)
            patient_hyper = np.sum(np.array(targets)[patient_mask])
            patient_stats.append({
                'patient_id': patient_id,
                'samples': patient_samples,
                'hyperglycemia_episodes': patient_hyper,
                'hyperglycemia_rate': patient_hyper / patient_samples if patient_samples > 0 else 0
            })
        
        stats_df = pd.DataFrame(patient_stats)
        
        # Distribuzione per paziente
        axes[0,1].hist(stats_df['hyperglycemia_rate'], bins=10, alpha=0.7, color='lightgreen')
        axes[0,1].set_title(f'Tasso di Iperglicemia per Paziente {title_suffix}')
        axes[0,1].set_xlabel('Tasso di Iperglicemia')
        axes[0,1].set_ylabel('Numero Pazienti')
        
        # Campioni per paziente
        axes[1,0].bar(range(len(stats_df)), stats_df['samples'], color='lightcoral')
        axes[1,0].set_title(f'Campioni per Paziente {title_suffix}')
        axes[1,0].set_xlabel('ID Paziente')
        axes[1,0].set_ylabel('Numero di Campioni')
        
        # Distribuzione glicemia
        all_glucose = np.array([seq['cgm_sequence'] for seq in sequences]).flatten()
        axes[1,1].hist(all_glucose, bins=50, alpha=0.7, color='gold')
        axes[1,1].axvline(x=180, color='red', linestyle='--', label='Soglia Iperglicemia')
        axes[1,1].set_title(f'Distribuzione Glicemia {title_suffix}')
        axes[1,1].set_xlabel('Glicemia (mg/dL)')
        axes[1,1].set_ylabel('Frequenza')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        # plt.show()
        
        return stats_df

    def load_and_preprocess(self):
        """Carica e preprocessa l'intero dataset HUPA-UCM"""
        print("Caricamento dataset HUPA-UCM...")
        
        patient_files = glob.glob(os.path.join(self.data_path, "HUPA*.csv"))
        patient_files.sort()
        
        if not patient_files:
            raise FileNotFoundError(f"Nessun file paziente trovato in {self.data_path}")
        
        print(f"Trovati {len(patient_files)} file pazienti")
        
        all_sequences = []
        successful_patients = 0
        
        for patient_file in patient_files:
            patient_id = os.path.basename(patient_file).replace('.csv', '').replace('HUPA', '').replace('P', '')
            
            patient_df = self.load_patient_data(patient_file)
            
            if patient_df is not None and len(patient_df) > self.sequence_length + self.prediction_horizon:
                patient_sequences = self.create_sequences(patient_df, patient_id)
                if patient_sequences:
                    all_sequences.extend(patient_sequences)
                    successful_patients += 1
                    print(f"Paziente {patient_id}: {len(patient_sequences)} sequenze create")
                else:
                    print(f"Paziente {patient_id}: nessuna sequenza valida")
            else:
                print(f"Paziente {patient_id}: dati insufficienti o corrotti")
        
        if not all_sequences:
            raise ValueError("Nessuna sequenza valida creata dal dataset")
        
        print(f"\nDataset processato: {len(all_sequences)} sequenze da {successful_patients} pazienti")
        
        # Analisi dati grezzi prima del preprocessing
        self.analyze_data_distribution(all_sequences, [seq['patient_id'] for seq in all_sequences])
        
        # Bilanciamento dei campioni per paziente
        print("\nBilanciamento campioni per paziente...")
        balanced_sequences, balanced_patient_ids = self.balance_patient_samples(
            all_sequences, 
            [seq['patient_id'] for seq in all_sequences],
            max_samples_per_patient=1000
        )
        
        print(f"Dataset dopo bilanciamento: {len(balanced_sequences)} sequenze")
        
        df_sequences = pd.DataFrame(balanced_sequences)
        
        # Estrazione features per il modello
        X_cgm = np.array(df_sequences['cgm_sequence'].tolist())
        X_fitbit = np.array(df_sequences['fitbit_sequence'].tolist())
        X_insulin = np.column_stack([
            df_sequences['total_bolus'].values,
            df_sequences['avg_basal'].values,
            df_sequences['total_carbs'].values
        ])
        y = df_sequences['target_hyperglycemia'].values
        patient_ids = df_sequences['patient_id'].values
        
        # Normalizzazione
        print("Normalizzazione features...")
        X_cgm_scaled = self.scaler_cgm.fit_transform(X_cgm)
        
        # Reshape per fitbit data
        X_fitbit_reshaped = X_fitbit.reshape(-1, X_fitbit.shape[-1])
        X_fitbit_scaled = self.scaler_fitbit.fit_transform(X_fitbit_reshaped)
        X_fitbit_scaled = X_fitbit_scaled.reshape(X_fitbit.shape)
        
        X_insulin_scaled = self.scaler_insulin.fit_transform(X_insulin)

        # Analisi dati dopo preprocessing
        self.analyze_data_distribution(balanced_sequences, balanced_patient_ids, is_raw_data=False)
        
        # Statistiche dataset
        print(f"\nStatistiche dataset:")
        print(f"- Campioni totali: {len(y)}")
        print(f"- Pazienti unici: {len(np.unique(patient_ids))}")
        print(f"- Episodi iperglicemia: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        print(f"- Distribuzione glicemia: {np.mean(X_cgm):.1f} ± {np.std(X_cgm):.1f} mg/dL")
        
        return {
            'cgm': X_cgm_scaled,
            'fitbit': X_fitbit_scaled,
            'insulin': X_insulin_scaled,
            'target': y,
            'patient_ids': patient_ids,
            'original_glucose': X_cgm
        }

    def balance_patient_samples(self, sequences, patient_ids, max_samples_per_patient=1000):
        """Bilancia il numero di campioni per paziente"""
        balanced_sequences = []
        balanced_patient_ids = []
        
        for patient_id in np.unique(patient_ids):
            patient_mask = np.array(patient_ids) == patient_id
            patient_indices = np.where(patient_mask)[0]
            
            # Se il paziente ha più campioni del massimo, fai campionamento casuale
            if len(patient_indices) > max_samples_per_patient:
                selected_indices = np.random.choice(
                    patient_indices, 
                    size=max_samples_per_patient, 
                    replace=False
                )
            else:
                selected_indices = patient_indices
            
            # Aggiungi campioni selezionati
            for idx in selected_indices:
                balanced_sequences.append(sequences[idx])
                balanced_patient_ids.append(patient_ids[idx])
        
        return balanced_sequences, balanced_patient_ids

class DiabetesMultiModalModel:
    """Modello multimodale per predizione iperglicemia"""
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def build_model(self, cgm_seq_length=96, fitbit_seq_length=96, insulin_features=3):
        """Costruisce l'architettura multimodale con Functional API"""
        
        # Input layers
        cgm_input = keras.Input(shape=(cgm_seq_length,), name='cgm_input')
        fitbit_input = keras.Input(shape=(fitbit_seq_length, 3), name='fitbit_input')
        insulin_input = keras.Input(shape=(insulin_features,), name='insulin_input')
        
        # Branch CGM
        cgm_reshaped = layers.Reshape((cgm_seq_length, 1))(cgm_input)
        cgm_lstm1 = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
        )(cgm_reshaped)
        cgm_attention = layers.Attention()([cgm_lstm1, cgm_lstm1])  # Self-attention
        cgm_lstm2 = layers.Bidirectional(
            layers.LSTM(32, dropout=0.2)
        )(cgm_attention)
        cgm_features = layers.Dense(32, activation='relu', name='cgm_features')(cgm_lstm2)
        
        # Branch Fitbit
        fitbit_normalized = layers.BatchNormalization()(fitbit_input)
        
        fitbit_lstm1 = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.1)
        )(fitbit_normalized)
        fitbit_attention = layers.Attention()([fitbit_lstm1, fitbit_lstm1])
        
        fitbit_lstm2 = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.1)
        )(fitbit_attention)
        
        # Global pooling per aggregare informazioni temporali
        fitbit_avg_pool = layers.GlobalAveragePooling1D()(fitbit_lstm2)
        fitbit_max_pool = layers.GlobalMaxPooling1D()(fitbit_lstm2)
        fitbit_pooled = layers.Concatenate()([fitbit_avg_pool, fitbit_max_pool])
        
        fitbit_dense1 = layers.Dense(64, activation='relu')(fitbit_pooled)
        fitbit_dropout1 = layers.Dropout(0.4)(fitbit_dense1)
        fitbit_features = layers.Dense(32, activation='relu', name='fitbit_features')(fitbit_dropout1)
        
        # Branch Insulin/Carb
        insulin_normalized = layers.BatchNormalization()(insulin_input)
        
        insulin_dense1 = layers.Dense(32, activation='relu')(insulin_normalized)
        insulin_dropout1 = layers.Dropout(0.4)(insulin_dense1)
        
        insulin_dense2 = layers.Dense(16, activation='relu')(insulin_dropout1)
        insulin_dropout2 = layers.Dropout(0.3)(insulin_dense2)
        
        insulin_dense3 = layers.Dense(8, activation='relu')(insulin_dropout2)
        insulin_features = layers.Dense(8, activation='relu', name='insulin_features')(insulin_dense3)
        
        # Fusion Layer        
        # 1. Concatenazione
        concat_features = layers.Concatenate(name='feature_fusion')([
            cgm_features, fitbit_features, insulin_features
        ])
        
        # 2. Feature interaction tramite dense layers
        # Proiezione delle singole modalità
        cgm_proj = layers.Dense(24, activation='relu')(cgm_features)
        fitbit_proj = layers.Dense(24, activation='relu')(fitbit_features)  
        insulin_proj = layers.Dense(24, activation='relu')(insulin_features)
        
        # Cross-modal interactions
        cgm_fitbit_interact = layers.Multiply()([cgm_proj, fitbit_proj])
        cgm_insulin_interact = layers.Multiply()([cgm_proj, insulin_proj])
        fitbit_insulin_interact = layers.Multiply()([fitbit_proj, insulin_proj])
        
        # Combinazione di tutte le interazioni
        interaction_features = layers.Concatenate()([
            cgm_fitbit_interact, cgm_insulin_interact, fitbit_insulin_interact
        ])
        
        # 3. Fusion finale
        fusion_input = layers.Concatenate()([
            concat_features,  # Features originali concatenate
            cgm_proj, fitbit_proj, insulin_proj,  # Features proiettate
            interaction_features  # Interazioni cross-modali
        ])
        
        # Classifier finale
        fusion_dense1 = layers.Dense(128, activation='relu')(fusion_input)
        fusion_dropout1 = layers.Dropout(0.5)(fusion_dense1)
        fusion_dense2 = layers.Dense(64, activation='relu')(fusion_dropout1)
        fusion_dropout2 = layers.Dropout(0.4)(fusion_dense2)
        fusion_dense3 = layers.Dense(32, activation='relu')(fusion_dropout2)
        fusion_dropout3 = layers.Dropout(0.3)(fusion_dense3)
        
        fusion_dense4 = layers.Dense(16, activation='relu')(fusion_dropout3)
        
        # 4. Output layer
        output = layers.Dense(1, activation='sigmoid', name='hyperglycemia_prediction')(fusion_dense4)
        
        # Modello finale
        self.model = keras.Model(
            inputs=[cgm_input, fitbit_input, insulin_input],
            outputs=output,
            name='diabetes_multimodal_model'
        )
        
        # Compilazione
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0008),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        print("\nArchitettura modello multimodale:")
        self.model.summary()

        plot_model(
            self.model, 
            to_file='plots/multimodal_model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            show_trainable=True,
            expand_nested=True,
            dpi=300,
            rankdir='TB'
        )
        
        return self.model
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Addestra il modello multimodale"""
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
        )
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            'models/best_diabetes_model.h5', monitor='val_auc', mode='max',
            save_best_only=True, verbose=1
        )
        
        print("Inizio training modello multimodale...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )

        self.plot_training_history()
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Valuta le prestazioni del modello"""
        
        # Predizioni
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Metriche
        test_metrics = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = self.model.metrics_names
        
        print("Risultati test set:")
        for i, metric_name in enumerate(metric_names):
            print(f"{metric_name.capitalize()}: {test_metrics[i]:.4f}")
        
        return {
            'metrics': dict(zip(metric_names, test_metrics)),
            'predictions': y_pred_proba,
            'binary_predictions': y_pred
        }
    
    def plot_training_history(self):
        """Visualizza le curve di training"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0,0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Loss
        axes[0,1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0,1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Precision
        axes[0,2].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[0,2].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[0,2].set_title('Model Precision')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Recall
        axes[1,0].plot(self.history.history['recall'], label='Train', linewidth=2)
        axes[1,0].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1,0].set_title('Model Recall')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # AUC
        axes[1,1].plot(self.history.history['auc'], label='Train', linewidth=2)
        axes[1,1].plot(self.history.history['val_auc'], label='Validation', linewidth=2)
        axes[1,1].set_title('Model AUC')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('AUC')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1,2].plot(self.history.history['lr'], linewidth=2)
            axes[1,2].set_title('Learning Rate')
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Learning Rate')
            axes[1,2].set_yscale('log')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png')
        # plt.show()

def print_training_summary(history, model_name):
    """Stampa il riepilogo del training per un determinato modello"""
    
    print(f"\n{model_name} Training Summary:")
    print(f"Training interrotto all'epoca {len(history.history['loss'])}")
    
    # Metriche finali
    print("\nMetriche finali:")
    print("Training:")
    print(f"- Loss: {history.history['loss'][-1]:.4f}")
    print(f"- Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"- AUC: {history.history['auc'][-1]:.4f}")
    print(f"- Precision: {history.history['precision'][-1]:.4f}")
    print(f"- Recall: {history.history['recall'][-1]:.4f}")
    
    print("\nValidation:")
    print(f"- Val Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"- Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"- Val AUC: {history.history['val_auc'][-1]:.4f}")
    print(f"- Val Precision: {history.history['val_precision'][-1]:.4f}")
    print(f"- Val Recall: {history.history['val_recall'][-1]:.4f}")
    
    # Migliori performance su validation
    best_epoch = np.argmin(history.history['val_loss'])
    print(f"\nMigliori performance (epoca {best_epoch + 1}):")
    print(f"- Best Val Loss: {min(history.history['val_loss']):.4f}")
    print(f"- Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"- Best Val AUC: {max(history.history['val_auc']):.4f}")
    print(f"- Best Val Precision: {max(history.history['val_precision']):.4f}")
    print(f"- Best Val Recall: {max(history.history['val_recall']):.4f}")
    
    print("\n" + "="*50)

def create_baseline_models(X_train, y_train, X_val, y_val):
    """Crea modelli baseline unimodali per confronto"""
    
    results = {
        'models': {},
        'metrics': {},
        'histories': {}
    }
    
    # Callbacks comuni, come il multimodale
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
    )

    # Training params comuni, come il multimodale
    training_params = {
        'epochs': 100,
        'batch_size': 32,
        'verbose': 1
    }

    # Baseline 1: CGM-only
    print("\nTraining baseline CGM-only...")
    cgm_input = keras.Input(shape=(96,))
    cgm_reshaped = layers.Reshape((96, 1))(cgm_input)
    cgm_lstm1 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(cgm_reshaped)
    cgm_attention = layers.Attention()([cgm_lstm1, cgm_lstm1])  # Self-attention
    cgm_lstm2 = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2)
    )(cgm_attention)
    cgm_features = layers.Dense(32, activation='relu')(cgm_lstm2)
    cgm_output = layers.Dense(1, activation='sigmoid')(cgm_features)
    
    cgm_model = keras.Model(cgm_input, cgm_output, name='cgm_baseline')
    cgm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    cgm_checkpoint = keras.callbacks.ModelCheckpoint(
        'models/best_cgm_baseline.h5', monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    )
    
    history_cgm = cgm_model.fit(
        X_train['cgm'], y_train,
        validation_data=(X_val['cgm'], y_val),
        callbacks=[early_stopping, reduce_lr],
        **training_params
    )
    print_training_summary(history_cgm, "CGM-only Model")
    cgm_metrics = cgm_model.evaluate(X_val['cgm'], y_val, verbose=0)
    
    results['models']['cgm'] = cgm_model
    results['metrics']['cgm'] = dict(zip(cgm_model.metrics_names, cgm_metrics))
    results['histories']['cgm'] = history_cgm



    # Baseline 2: Fitbit-only
    print("\nTraining baseline Fitbit-only...")
    fitbit_input = keras.Input(shape=(96, 3))
    
    fitbit_normalized = layers.BatchNormalization()(fitbit_input)
    
    fitbit_lstm1 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.1)
    )(fitbit_normalized)
    fitbit_attention = layers.Attention()([fitbit_lstm1, fitbit_lstm1])
    
    fitbit_lstm2 = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.1)
    )(fitbit_attention)
    
    # Global pooling per aggregare informazioni temporali
    fitbit_avg_pool = layers.GlobalAveragePooling1D()(fitbit_lstm2)
    fitbit_max_pool = layers.GlobalMaxPooling1D()(fitbit_lstm2)
    fitbit_pooled = layers.Concatenate()([fitbit_avg_pool, fitbit_max_pool])
    
    fitbit_dense1 = layers.Dense(64, activation='relu')(fitbit_pooled)
    fitbit_dropout1 = layers.Dropout(0.4)(fitbit_dense1)
    fitbit_features = layers.Dense(32, activation='relu')(fitbit_dropout1)
    fitbit_dropout2 = layers.Dropout(0.3)(fitbit_features)
    fitbit_output = layers.Dense(1, activation='sigmoid')(fitbit_dropout2)
    
    fitbit_model = keras.Model(fitbit_input, fitbit_output, name='fitbit_baseline')
    fitbit_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    fitbit_checkpoint = keras.callbacks.ModelCheckpoint(
        'models/best_fitbit_baseline.h5', monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    )
    
    history_fitbit = fitbit_model.fit(
        X_train['fitbit'], y_train,
        validation_data=(X_val['fitbit'], y_val),
        callbacks=[early_stopping, reduce_lr],
        **training_params
    )
    print_training_summary(history_fitbit, "Fitbit-only Model")
    fitbit_metrics = fitbit_model.evaluate(X_val['fitbit'], y_val, verbose=0)
    
    results['models']['fitbit'] = fitbit_model
    results['metrics']['fitbit'] = dict(zip(fitbit_model.metrics_names, fitbit_metrics))
    results['histories']['fitbit'] = history_fitbit



    # Baseline 3: Insulin-only
    print("\nTraining baseline Insulin-only...")
    insulin_input = keras.Input(shape=(3,))
    
    insulin_normalized = layers.BatchNormalization()(insulin_input)
    
    insulin_dense1 = layers.Dense(32, activation='relu')(insulin_normalized)
    insulin_dropout1 = layers.Dropout(0.4)(insulin_dense1)
    
    insulin_dense2 = layers.Dense(16, activation='relu')(insulin_dropout1)
    insulin_dropout2 = layers.Dropout(0.3)(insulin_dense2)
    
    insulin_dense3 = layers.Dense(8, activation='relu')(insulin_dropout2)
    insulin_features = layers.Dense(8, activation='relu')(insulin_dense3)
    
    insulin_dropout3 = layers.Dropout(0.2)(insulin_features)
    insulin_output = layers.Dense(1, activation='sigmoid')(insulin_dropout3)
    
    insulin_model = keras.Model(insulin_input, insulin_output, name='insulin_baseline')
    insulin_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    insulin_checkpoint = keras.callbacks.ModelCheckpoint(
        'models/best_insulin_baseline.h5', monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    )
    
    history_insulin = insulin_model.fit(
        X_train['insulin'], y_train,
        validation_data=(X_val['insulin'], y_val),
        callbacks=[early_stopping, reduce_lr],
        **training_params
    )
    print_training_summary(history_insulin, "Insulin-only Model")
    insulin_metrics = insulin_model.evaluate(X_val['insulin'], y_val, verbose=0)
    
    results['models']['insulin'] = insulin_model
    results['metrics']['insulin'] = dict(zip(insulin_model.metrics_names, insulin_metrics))
    results['histories']['insulin'] = history_insulin

    

    return results
    

def main():    
    try:
        data_loader = DiabetesMultiModalDataLoader("dataset/preprocessed")
        data = data_loader.load_and_preprocess()

        train_patients, val_patients, test_patients = data_loader.train_val_test_split(data)    
        X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_data(data, train_patients, val_patients, test_patients)    
    except Exception as e:
        print(f"Errore nel caricamento dataset: {e}")
        return
    
    try:
        multimodal_model = DiabetesMultiModalModel()
        model = multimodal_model.build_model()
    except Exception as e:
        print(f"Errore nella costruzione del modello: {e}")
        return
    
    try:
        history = multimodal_model.train(
            [X_train['cgm'], X_train['fitbit'], X_train['insulin']], y_train,
            [X_val['cgm'], X_val['fitbit'], X_val['insulin']], y_val,
            epochs=100, batch_size=32
        )
        print_training_summary(history, "Multimodal Model")
    except Exception as e:
        print(f"Errore nel training del modello: {e}")
        return
    
    try:
        baseline_results = create_baseline_models(X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"Errore nella creazione dei modelli baseline: {e}")
        return
    
    # Valutazione comparativa
    try:
        # Modello multimodale
        print("\nMODELLO MULTIMODALE:")
        multimodal_results = multimodal_model.evaluate(
            [X_test['cgm'], X_test['fitbit'], X_test['insulin']], y_test
        )
        
        # Baseline CGM
        print("\nBASELINE CGM-only:")
        cgm_metrics = baseline_results['metrics']['cgm']
        for i, metric_name in enumerate(cgm_metrics.keys()):
            print(f"{metric_name.capitalize()}: {cgm_metrics[metric_name]:.4f}")
        
        # Baseline Fitbit
        print("\nBASELINE Fitbit-only:")
        fitbit_metrics = baseline_results['metrics']['fitbit']
        for i, metric_name in enumerate(fitbit_metrics.keys()):
            print(f"{metric_name.capitalize()}: {fitbit_metrics[metric_name]:.4f}")
        
        # Baseline Insulin
        print("\nBASELINE Insulin-only:")
        insulin_metrics = baseline_results['metrics']['insulin']
        for i, metric_name in enumerate(insulin_metrics.keys()):
            print(f"{metric_name.capitalize()}: {insulin_metrics[metric_name]:.4f}")


        # Confronto prestazioni
        comparison_data = {
            'Model': ['Multimodal', 'CGM-only', 'Fitbit-only', 'Insulin-only'],
            'Accuracy': [
                multimodal_results['metrics']['accuracy'],
                cgm_metrics['accuracy'],
                fitbit_metrics['accuracy'],
                insulin_metrics['accuracy']
            ],
            'AUC': [
                multimodal_results['metrics']['auc'],
                cgm_metrics['auc'],
                fitbit_metrics['auc'],
                insulin_metrics['auc']
            ],
            'Precision': [
                multimodal_results['metrics']['precision'],
                cgm_metrics['precision'],
                fitbit_metrics['precision'],
                insulin_metrics['precision']
            ],
            'Recall': [
                multimodal_results['metrics']['recall'],
                cgm_metrics['recall'],
                fitbit_metrics['recall'],
                insulin_metrics['recall']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Miglior modello
        best_model_idx = np.argmax(comparison_df['AUC'])
        best_model = comparison_df.iloc[best_model_idx]['Model']
        print(f"\nMiglior modello: {best_model}")
    except Exception as e:
        print(f"Errore nella valutazione comparativa: {e}")
        return


if __name__ == "__main__":
    main()