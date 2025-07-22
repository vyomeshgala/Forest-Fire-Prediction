import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import joblib
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings('ignore')


class WildfirePredictionModel:
    """
    A comprehensive machine learning model for predicting wildfire likelihood
    using historical data with geographical visualization capabilities.
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.class_weights = None
        self.best_model = None
        self.best_model_name = None

    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic wildfire data for demonstration purposes.
        In a real scenario, this would be replaced with actual historical data loading.
        """
        np.random.seed(42)

        # Generate features that could influence wildfire occurrence
        data = {
            # Weather conditions
            'temperature': np.random.normal(25, 10, n_samples),  # Celsius
            'humidity': np.random.uniform(10, 90, n_samples),  # Percentage
            'wind_speed': np.random.exponential(15, n_samples),  # km/h
            'precipitation': np.random.exponential(2, n_samples),  # mm

            # Geographical features
            'latitude': np.random.uniform(30, 50, n_samples),
            'longitude': np.random.uniform(-120, -80, n_samples),
            'elevation': np.random.uniform(0, 3000, n_samples),  # meters
            'slope': np.random.uniform(0, 45, n_samples),  # degrees

            # Vegetation and land use
            'vegetation_density': np.random.uniform(0, 100, n_samples),  # Percentage
            'drought_index': np.random.uniform(0, 4, n_samples),  # Palmer Drought Severity Index

            # Seasonal factors
            'month': np.random.randint(1, 13, n_samples),
            'day_of_year': np.random.randint(1, 366, n_samples),

            # Human factors
            'population_density': np.random.exponential(50, n_samples),  # people per sq km
            'distance_to_road': np.random.exponential(5, n_samples),  # km
        }

        # Create realistic wildfire probability based on conditions
        fire_probability = (
                0.3 * np.where(data['temperature'] > 30, 1, 0) +  # High temperature
                0.2 * np.where(data['humidity'] < 30, 1, 0) +  # Low humidity
                0.2 * np.where(data['wind_speed'] > 20, 1, 0) +  # High wind
                0.1 * np.where(data['precipitation'] < 1, 1, 0) +  # Low precipitation
                0.1 * np.where(data['drought_index'] > 2, 1, 0) +  # Drought conditions
                0.05 * np.where(np.isin(data['month'], [6, 7, 8, 9]), 1, 0) +  # Fire season
                0.05 * np.where(data['vegetation_density'] > 70, 1, 0)  # Dense vegetation
        )

        # Add some noise and create binary target
        fire_probability += np.random.normal(0, 0.1, n_samples)
        data['wildfire_occurred'] = np.where(fire_probability > 0.4, 1, 0)

        # Create imbalanced dataset (wildfires are rare events)
        fire_indices = np.where(data['wildfire_occurred'] == 1)[0]
        no_fire_indices = np.where(data['wildfire_occurred'] == 0)[0]

        # Keep only 15% as fire events to simulate real imbalance
        n_fire_keep = min(int(len(fire_indices) * 0.15), len(fire_indices))
        n_no_fire_keep = min(n_samples - n_fire_keep, len(no_fire_indices))

        # Ensure we don't exceed available samples
        if n_fire_keep > len(fire_indices):
            n_fire_keep = len(fire_indices)
        if n_no_fire_keep > len(no_fire_indices):
            n_no_fire_keep = len(no_fire_indices)

        selected_fire_indices = np.random.choice(fire_indices, n_fire_keep, replace=False)
        selected_no_fire_indices = np.random.choice(no_fire_indices, n_no_fire_keep, replace=False)

        selected_indices = np.concatenate([selected_fire_indices, selected_no_fire_indices])

        for key in data.keys():
            data[key] = data[key][selected_indices]

        return pd.DataFrame(data)

    def load_and_preprocess_data(self, data_path=None):
        """
        Load and preprocess the wildfire data.
        If data_path is None, generate synthetic data for demonstration.
        """
        if data_path is None:
            print("Generating synthetic wildfire data for demonstration...")
            self.df = self.generate_synthetic_data()
        else:
            print(f"Loading data from {data_path}...")
            self.df = pd.read_csv(data_path)

        print(f"Dataset shape: {self.df.shape}")
        print(f"Wildfire occurrence distribution:")
        print(self.df['wildfire_occurred'].value_counts(normalize=True))

        # Handle missing values
        self.df = self.df.fillna(self.df.median())

        # Feature engineering
        self.df['temp_humidity_ratio'] = self.df['temperature'] / (self.df['humidity'] + 1)
        self.df['fire_weather_index'] = (
                self.df['temperature'] * self.df['wind_speed'] / (self.df['humidity'] + 1)
        )
        self.df['seasonal_risk'] = np.where(self.df['month'].isin([6, 7, 8, 9]), 1, 0)

        # Prepare features and target
        feature_cols = [col for col in self.df.columns if col != 'wildfire_occurred']
        self.feature_names = feature_cols

        X = self.df[feature_cols]
        y = self.df['wildfire_occurred']

        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        return X_scaled, y

    def handle_class_imbalance(self, X, y, method='smote'):
        """
        Handle class imbalance using various techniques.
        """
        print(f"Original class distribution: {np.bincount(y)}")

        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
        elif method == 'class_weight':
            # Calculate class weights
            classes = np.unique(y)
            self.class_weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = dict(zip(classes, self.class_weights))
            return X, y
        else:
            return X, y

        print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        return X_balanced, y_balanced

    def train_models(self, X, y, use_class_weights=False):
        """
        Train multiple models with cross-validation.
        """
        # Define models
        if use_class_weights and self.class_weights:
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight=self.class_weights
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42,
                    class_weight=self.class_weights,
                    max_iter=1000
                )
            }
        else:
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                )
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        results = {}

        print("Training and evaluating models with cross-validation...")
        print("-" * 60)

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                        scoring='roc_auc', n_jobs=-1)

            # Train on full training set
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)

            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_auc': auc_score,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test AUC: {auc_score:.4f}")

        self.models = results

        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name

        print(f"\nBest model: {best_model_name}")

        return X_train, X_test, y_train, y_test

    def evaluate_models(self):
        """
        Detailed evaluation of all trained models without plotting.
        """
        print("\nModel Evaluation Results:")
        print("=" * 60)

        for name in self.models.keys():
            model_results = self.models[name]
            print(f"\n{name}:")
            print(f"  Cross-validation AUC: {model_results['cv_mean']:.4f} (+/- {model_results['cv_std'] * 2:.4f})")
            print(f"  Test AUC: {model_results['test_auc']:.4f}")

        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Cross-validation AUC: {self.models[self.best_model_name]['cv_mean']:.4f}")

        # Feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\nTop 10 Feature Importances - {self.best_model_name}:")
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]

            for i, idx in enumerate(indices):
                print(f"  {i + 1:2d}. {self.feature_names[idx]:20s} - {importance[idx]:.4f}")

        # Confusion matrix for best model
        y_test = self.models[self.best_model_name]['y_test']
        y_pred = self.models[self.best_model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)

        print(f"\nConfusion Matrix - {self.best_model_name}:")
        print("                 Predicted")
        print("                No Fire  Fire")
        print(f"Actual No Fire     {cm[0, 0]:4d}  {cm[0, 1]:4d}")
        print(f"       Fire        {cm[1, 0]:4d}  {cm[1, 1]:4d}")

        # Print detailed classification report
        print(f"\nClassification Report - {self.best_model_name}:")
        print("-" * 50)
        print(classification_report(y_test, y_pred))

    def create_geographical_visualization(self):
        """
        Create geographical visualizations for QGIS integration.
        """
        # Create a sample of predictions for visualization
        sample_size = min(1000, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        sample_df = self.df.iloc[sample_indices].copy()

        # Get predictions for sample
        X_sample = sample_df[self.feature_names]
        X_sample_scaled = self.scaler.transform(X_sample)
        predictions = self.best_model.predict_proba(X_sample_scaled)[:, 1]

        sample_df['fire_risk_prob'] = predictions  # Shortened for shapefile compatibility
        sample_df['risk_cat'] = pd.cut(predictions,
                                       bins=[0, 0.3, 0.6, 1.0],
                                       labels=['Low', 'Medium', 'High'])

        # Create risk score for continuous visualization
        sample_df['risk_score'] = (predictions * 100).round(1)  # 0-100 scale

        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(sample_df['longitude'], sample_df['latitude'])]
        gdf = gpd.GeoDataFrame(sample_df, geometry=geometry)

        # Set coordinate reference system (WGS84)
        gdf.crs = "EPSG:4326"

        # Save to multiple formats for QGIS
        print("Creating geographical visualization files...")

        # 1. Shapefile (most common)
        gdf.to_file('wildfire_predictions.shp')

        # 2. GeoJSON (web-friendly)
        gdf.to_file('wildfire_predictions.geojson', driver='GeoJSON')

        # 3. CSV with coordinates (for simple import)
        csv_df = sample_df[['longitude', 'latitude', 'fire_risk_prob', 'risk_cat',
                            'temperature', 'humidity', 'wind_speed', 'precipitation',
                            'elevation', 'slope', 'vegetation_density', 'drought_index']].copy()
        csv_df.to_csv('wildfire_predictions.csv', index=False)

        # 4. Create a raster-like grid for interpolation (optional)
        self.create_risk_grid(sample_df)

        # 5. Generate QGIS style file (.qml)
        self.create_qgis_style_file()

        print("Geographical visualization files created:")
        print("- wildfire_predictions.shp (Shapefile for QGIS)")
        print("- wildfire_predictions.geojson (GeoJSON format)")
        print("- wildfire_predictions.csv (CSV with coordinates)")
        print("- wildfire_risk_grid.csv (Grid for interpolation)")
        print("- wildfire_style.qml (QGIS style file)")

        return gdf

    def create_risk_grid(self, sample_df):
        """
        Create a regular grid for raster interpolation in QGIS.
        """
        # Define grid bounds
        min_lon, max_lon = sample_df['longitude'].min(), sample_df['longitude'].max()
        min_lat, max_lat = sample_df['latitude'].min(), sample_df['latitude'].max()

        # Create grid
        lon_range = np.linspace(min_lon, max_lon, 50)
        lat_range = np.linspace(min_lat, max_lat, 50)

        grid_data = []
        for lon in lon_range:
            for lat in lat_range:
                # Create synthetic data point for grid cell
                grid_point = {
                    'longitude': lon,
                    'latitude': lat,
                    'temperature': sample_df['temperature'].mean(),
                    'humidity': sample_df['humidity'].mean(),
                    'wind_speed': sample_df['wind_speed'].mean(),
                    'precipitation': sample_df['precipitation'].mean(),
                    'elevation': sample_df['elevation'].mean(),
                    'slope': sample_df['slope'].mean(),
                    'vegetation_density': sample_df['vegetation_density'].mean(),
                    'drought_index': sample_df['drought_index'].mean(),
                    'month': 7,  # Peak fire season
                    'day_of_year': 200,
                    'population_density': sample_df['population_density'].mean(),
                    'distance_to_road': sample_df['distance_to_road'].mean(),
                }

                # Add engineered features
                grid_point['temp_humidity_ratio'] = grid_point['temperature'] / (grid_point['humidity'] + 1)
                grid_point['fire_weather_index'] = (grid_point['temperature'] * grid_point['wind_speed'] /
                                                    (grid_point['humidity'] + 1))
                grid_point['seasonal_risk'] = 1

                grid_data.append(grid_point)

        grid_df = pd.DataFrame(grid_data)

        # Predict risk for grid points
        X_grid = grid_df[self.feature_names]
        X_grid_scaled = self.scaler.transform(X_grid)
        grid_predictions = self.best_model.predict_proba(X_grid_scaled)[:, 1]

        grid_df['fire_risk_prob'] = grid_predictions
        grid_df['x'] = grid_df['longitude']
        grid_df['y'] = grid_df['latitude']
        grid_df['z'] = grid_df['fire_risk_prob']

        # Save grid for QGIS interpolation
        grid_df[['x', 'y', 'z']].to_csv('wildfire_risk_grid.csv', index=False)

    def create_qgis_style_file(self):
        """
        Create a QGIS style file (.qml) for automatic styling.
        """
        qml_content = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.0">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" alphaBand="-1" type="singlebandpseudocolor" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
          <colorramp type="gradient" name="[source]">
            <prop k="color1" v="0,255,0,255"/>
            <prop k="color2" v="255,0,0,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.33;255,255,0,255:0.66;255,165,0,255"/>
          </colorramp>
          <item alpha="255" value="0" label="Low Risk (0.0)" color="#00ff00"/>
          <item alpha="255" value="0.3" label="Low-Medium (0.3)" color="#ffff00"/>
          <item alpha="255" value="0.6" label="Medium-High (0.6)" color="#ffa500"/>
          <item alpha="255" value="1" label="High Risk (1.0)" color="#ff0000"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>'''

        with open('wildfire_style.qml', 'w') as f:
            f.write(qml_content)

    def save_model(self, filepath='wildfire_model.pkl'):
        """
        Save the trained model and preprocessing components.
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='wildfire_model.pkl'):
        """
        Load a previously trained model.
        """
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.class_weights = model_data.get('class_weights', None)
        print(f"Model loaded from {filepath}")

    def predict_wildfire_risk(self, new_data):
        """
        Predict wildfire risk for new data points.
        """
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Scale the data
        X_new = new_data[self.feature_names]
        X_new_scaled = self.scaler.transform(X_new)

        # Make predictions
        risk_probability = self.best_model.predict_proba(X_new_scaled)[:, 1]
        risk_category = pd.cut(risk_probability,
                               bins=[0, 0.3, 0.6, 1.0],
                               labels=['Low', 'Medium', 'High'])

        results = pd.DataFrame({
            'fire_risk_probability': risk_probability,
            'risk_category': risk_category
        })

        return results


def main():
    """
    Main function to demonstrate the wildfire prediction model.
    """
    # Initialize the model
    wildfire_model = WildfirePredictionModel()

    # Load and preprocess data
    X, y = wildfire_model.load_and_preprocess_data()

    # Handle class imbalance using SMOTE
    X_balanced, y_balanced = wildfire_model.handle_class_imbalance(X, y, method='smote')

    # Train models
    X_train, X_test, y_train, y_test = wildfire_model.train_models(X_balanced, y_balanced)

    # Evaluate models
    wildfire_model.evaluate_models()

    # Create geographical visualizations
    gdf = wildfire_model.create_geographical_visualization()

    # Save the model
    wildfire_model.save_model('wildfire_prediction_model.pkl')

    # Example prediction for new data
    print("\nExample prediction for new data:")
    new_data = {
        'temperature': 35,
        'humidity': 20,
        'wind_speed': 25,
        'precipitation': 0.5,
        'latitude': 40.0,
        'longitude': -100.0,
        'elevation': 1000,
        'slope': 15,
        'vegetation_density': 80,
        'drought_index': 3.0,
        'month': 7,
        'day_of_year': 200,
        'population_density': 10,
        'distance_to_road': 2,
        'temp_humidity_ratio': 35 / 21,
        'fire_weather_index': 35 * 25 / 21,
        'seasonal_risk': 1
    }

    prediction = wildfire_model.predict_wildfire_risk(new_data)
    print(f"Fire risk probability: {prediction['fire_risk_probability'].iloc[0]:.3f}")
    print(f"Risk category: {prediction['risk_category'].iloc[0]}")

    return wildfire_model


if __name__ == "__main__":
    model = main()