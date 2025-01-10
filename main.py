import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Assignment5:

    # Class initializer
    def __init__(self, X, y):

        # Split training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initailize models
        self.models = {
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(),
            'knn': KNeighborsClassifier(),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(probability=True)
        }

        # Initailize feature extractor
        self.feature_extractor = PCA(n_components=0.9)

    def data_preprocessing(self):

        # Transform data from string labels to integer labels
        self.y_train = LabelEncoder().fit_transform(self.y_train)
        self.y_test = LabelEncoder().fit_transform(self.y_test)

    def training(self):

         # Iterate all 3 models, fit them on the training data, then print
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            print(f"{model_name} model trained.")

    def testing(self):

        # evaluate the model on test set and get their performance
        for model_name, model in self.models.items():
            accuracy = model.score(self.X_test, self.y_test)
            print(f"{model_name} Testing Accuracy: {accuracy:.2f}")
        final_predictions = self.weighted_scoring(self.X_test)
        final_accuracy = accuracy_score(self.y_test, final_predictions)
        print(f"Final Model Testing Accuracy: {final_accuracy:.2f}")

    def feature_extraction(self):

        # Dimension reduction to make the algorithm faster, the Pov is 0.9
        self.X_train = self.feature_extractor.fit_transform(self.X_train)
        self.X_test = self.feature_extractor.transform(self.X_test)


    def calculate_loss(self, weights, X_val, y_val):
        """Calculate loss function, e.g., negative accuracy."""
        weighted_predictions = self.weighted_scoring(X_val, weights)
        accuracy = accuracy_score(y_val, weighted_predictions)
        return -accuracy  # Negative accuracy as we want to maximize accuracy

    def gradient_descent(self, X_val, y_val, learning_rate=0.01, iterations=100):
        """Simple gradient descent to optimize weights."""
        # Initialize weights
        weights = np.ones(len(self.models)) / len(self.models)

        for _ in range(iterations):
            # Calculate gradients (here, using finite difference method)
            gradients = np.zeros(len(weights))
            for i in range(len(weights)):
                weights[i] += 0.001  # Small change in weight
                loss_plus = self.calculate_loss(weights, X_val, y_val)
                weights[i] -= 0.002  # Small change in the opposite direction
                loss_minus = self.calculate_loss(weights, X_val, y_val)
                gradients[i] = (loss_plus - loss_minus) / 0.002  # Finite difference
                weights[i] += 0.001  # Reset weight

            # Update weights
            weights -= learning_rate * gradients

        return weights

    def weighted_scoring(self, X, weights=None):
        if weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
        predictions = np.array([model.predict_proba(X)[:, 1] for _, model in self.models.items()])
        weighted_predictions = np.dot(predictions.T, weights)
        return (weighted_predictions >= 0.5).astype(int)

    def cross_validation_with_optimization(self, k=5):
        """Performs K-Fold cross-validation and optimizes weights."""
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_weights = []

        for train_index, val_index in skf.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train[train_index], self.X_train[val_index]
            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

            # Model training
            for _, model in self.models.items():
                model.fit(X_train_fold, y_train_fold)

            # Gradient descent
            optimal_weights = self.gradient_descent(X_val_fold, y_val_fold)
            fold_weights.append(optimal_weights)

        # Average folds
        average_weights = np.mean(fold_weights, axis=0)
        print(f"Average Optimal weights across folds: {average_weights}")
    
def simplify_stages(y):
    
    # This function will take the tumor stage labels and simplify them
    # 'stage ia', 'stage ib' -> 'stage i'; stage iia', 'stage iib' -> 'stage ii', etc.
    simplified_y = y.str.extract(r'(stage [ivx]+)', expand=False)
    return simplified_y

def extract_data(file_path, column_name, num):
    data = pd.read_csv(file_path, sep='\t')
    data = data.set_index([column_name]).T.reset_index().rename(columns={'index': 'sample_id'})
    #print(1)

    sample_ids = data['sample_id']
    #print(2)

    data = data.dropna(axis=1, how='all')
    data_numeric = data.drop('sample_id', axis=1)
    #print(3)

    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_numeric)
    #print(data_imputed.shape)

    # PCA
    feature_extractor = PCA(n_components=num)
    data_transformed = feature_extractor.fit_transform(data_imputed)

    data_transformed = pd.DataFrame(data_transformed, index=sample_ids)
    #print(data_transformed)

    return data_transformed
    
def main():

    # Access to the table
    counts = extract_data('TCGA-BRCA.htseq_counts.tsv', 'Ensembl_ID', 800)
    mirna = extract_data('TCGA-BRCA.mirna.tsv', 'miRNA_ID',90)
    methylation1 = extract_data('TCGA-BRCA.methylation450.tsv','Composite Element REF',600)
    #methylation2 = extract_data('TCGA-BRCA.methylation27.tsv','Composite Element REF',200)
    #methylation_combined = pd.concat([methylation1, methylation2], axis=1)



    #methylation27 = pd.read_csv('TCGA-BRCA.methylation27.tsv', 'Composite Element REF', 60)
    
    stage = pd.read_csv('TCGA-BRCA.GDC_phenotype.tsv', sep='\t')
    # Change the column name for merging
    stage1 = stage.rename(columns={'submitter_id.samples': 'sample_id'})
    
    merged = pd.merge(counts, mirna, on='sample_id', how='inner')
    merged = pd.merge(merged, methylation1, on='sample_id', how='inner')
    #merged = pd.merge(merged, methylation2, on='sample_id', how='inner')
    merged = pd.merge(merged, stage1, on='sample_id', how='inner')

    merged['tumor_stage.diagnoses'] = simplify_stages(merged['tumor_stage.diagnoses'])

    # Separating features and labels
    X = merged.drop(['sample_id', 'tumor_stage.diagnoses'], axis=1)
    y = merged['tumor_stage.diagnoses']

    X.columns = X.columns.astype(str)

    A5 = Assignment5(X, y)
    A5.data_preprocessing()
    A5.feature_extraction()
    A5.training()
    A5.cross_validation_with_optimization(k=5)
    A5.testing()

    
    print('Hahaha, I finished assignment5.')

if __name__ == "__main__":
    main()

