#!/usr/bin/env python3
"""
Enhanced ML Performance Predictor with Local vs Cloud Recommendations
Run this file directly: python complete_predictor.py
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import json
import pickle
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import re
import os
from pathlib import Path
import time
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkData:
    """Structure for benchmark data points"""
    model_name: str
    dataset: str
    gpu_type: str
    cpu_type: str
    batch_size: int
    epochs: int
    dataset_size: int
    model_params: int
    flops: float
    memory_gb: float
    training_time: float  # hours
    energy_usage: float   # kWh
    cost_usd: float      # USD
    framework: str
    precision: str
    timestamp: datetime

@dataclass
class HardwareSpec:
    """Local hardware specification"""
    gpu_type: str
    gpu_memory_gb: float
    cpu_type: str
    ram_gb: float
    storage_gb: float
    power_cost_per_kwh: float = 0.12  # USD per kWh
    
@dataclass
class TrainingRecommendation:
    """Training recommendation result"""
    recommendation: str  # "local" or "cloud"
    confidence: float   # 0-1
    local_prediction: Dict[str, float]
    cloud_prediction: Dict[str, float]
    reasoning: List[str]
    cost_savings: float
    time_difference: float
    feasibility_score: float

class BenchmarkScraper:
    """Scrapes ML benchmark data from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def scrape_mlperf_data(self) -> List[BenchmarkData]:
        """Scrape MLPerf benchmark results"""
        benchmarks = []
        try:
            # For demonstration, using synthetic data
            logger.info("Generating synthetic MLPerf-style data...")
            benchmarks.extend(self._get_synthetic_mlperf_data())
            
        except Exception as e:
            logger.error(f"Failed to scrape MLPerf data: {e}")
            benchmarks.extend(self._get_synthetic_mlperf_data())
            
        return benchmarks
    
    def _get_synthetic_mlperf_data(self) -> List[BenchmarkData]:
        """Generate synthetic MLPerf-style data for demonstration"""
        models = [
            ("ResNet-50", "ImageNet", 25600000, 4.1e9),
            ("BERT-Large", "Wikipedia", 340000000, 1.3e11),
            ("GPT-2", "OpenWebText", 1500000000, 3.1e12),
            ("T5-Large", "C4", 770000000, 2.8e11),
            ("ViT-Large", "ImageNet", 307000000, 6.1e10)
        ]
        
        gpus = [
            ("A100", 40, 400, 3.0),
            ("V100", 32, 300, 2.0),
            ("RTX 3090", 24, 220, 1.5),
            ("H100", 80, 700, 5.0),
            ("RTX 4090", 24, 330, 2.5)
        ]
        
        benchmarks = []
        for model_name, dataset, params, flops in models:
            for gpu_name, memory, power_watts, price_per_hour in gpus:
                for batch_size in [8, 16, 32]:
                    # Simulate training time based on model complexity and hardware
                    base_time = (params / 1e8) * (flops / 1e12) / (memory / 10) * (64 / batch_size)
                    training_time = max(0.5, base_time * np.random.uniform(0.8, 1.2))
                    
                    benchmark = BenchmarkData(
                        model_name=model_name,
                        dataset=dataset,
                        gpu_type=gpu_name,
                        cpu_type="Intel Xeon",
                        batch_size=batch_size,
                        epochs=100,
                        dataset_size=self._estimate_dataset_size(dataset),
                        model_params=params,
                        flops=flops,
                        memory_gb=memory,
                        training_time=training_time,
                        energy_usage=training_time * (power_watts / 1000),
                        cost_usd=training_time * price_per_hour,
                        framework=np.random.choice(["PyTorch", "TensorFlow"]),
                        precision=np.random.choice(["FP16", "FP32"]),
                        timestamp=datetime.now()
                    )
                    benchmarks.append(benchmark)
                    
        return benchmarks
    
    def _estimate_dataset_size(self, dataset: str) -> int:
        """Estimate dataset size"""
        size_map = {
            'imagenet': 1281167,
            'coco': 118287,
            'wikipedia': 6000000,
            'openwebtext': 40000000,
            'c4': 365000000
        }
        
        for key, size in size_map.items():
            if key in dataset.lower():
                return size
        return 100000

class DataIngestion:
    """Handle ingestion of external benchmark data"""
    
    @staticmethod
    def ingest_csv(file_path: str) -> List[BenchmarkData]:
        """Ingest benchmark data from CSV"""
        benchmarks = []
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loading CSV with {len(df)} records from {file_path}")
            
            for _, row in df.iterrows():
                benchmark = BenchmarkData(
                    model_name=str(row.get('model_name', 'Unknown')),
                    dataset=str(row.get('dataset', 'Unknown')),
                    gpu_type=str(row.get('gpu_type', 'Unknown')),
                    cpu_type=str(row.get('cpu_type', 'Unknown')),
                    batch_size=int(row.get('batch_size', 32)),
                    epochs=int(row.get('epochs', 100)),
                    dataset_size=int(row.get('dataset_size', 100000)),
                    model_params=int(row.get('model_params', 50000000)),
                    flops=float(row.get('flops', 1e10)),
                    memory_gb=float(row.get('memory_gb', 16)),
                    training_time=float(row.get('training_time', 1.0)),
                    energy_usage=float(row.get('energy_usage', 0.5)),
                    cost_usd=float(row.get('cost_usd', 2.0)),
                    framework=str(row.get('framework', 'PyTorch')),
                    precision=str(row.get('precision', 'FP32')),
                    timestamp=datetime.now()
                )
                benchmarks.append(benchmark)
                
        except Exception as e:
            logger.error(f"Failed to ingest CSV {file_path}: {e}")
            
        return benchmarks

class FeatureEngineer:
    """Extract and engineer features for ML training"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def extract_features(self, benchmarks: List[BenchmarkData]) -> pd.DataFrame:
        """Extract features from benchmark data"""
        features = []
        
        for benchmark in benchmarks:
            feature_dict = {
                # Basic features
                'model_params': benchmark.model_params,
                'flops': benchmark.flops,
                'dataset_size': benchmark.dataset_size,
                'batch_size': benchmark.batch_size,
                'epochs': benchmark.epochs,
                'memory_gb': benchmark.memory_gb,
                
                # Categorical features (will be encoded)
                'model_name': benchmark.model_name,
                'dataset': benchmark.dataset,
                'gpu_type': benchmark.gpu_type,
                'cpu_type': benchmark.cpu_type,
                'framework': benchmark.framework,
                'precision': benchmark.precision,
                
                # Derived features
                'flops_per_param': benchmark.flops / max(benchmark.model_params, 1),
                'params_per_batch': benchmark.model_params / max(benchmark.batch_size, 1),
                'dataset_batches': benchmark.dataset_size / max(benchmark.batch_size, 1),
                'compute_intensity': benchmark.flops / max(benchmark.memory_gb, 1),
                'memory_per_param': benchmark.memory_gb * 1e9 / max(benchmark.model_params, 1),
                
                # Targets
                'training_time': benchmark.training_time,
                'energy_usage': benchmark.energy_usage,
                'cost_usd': benchmark.cost_usd
            }
            features.append(feature_dict)
            
        df = pd.DataFrame(features)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Log transform skewed features
        skewed_features = ['model_params', 'flops', 'dataset_size', 'flops_per_param']
        for feature in skewed_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(df[feature])
                
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_features = ['model_name', 'dataset', 'gpu_type', 'cpu_type', 'framework', 'precision']
        
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    df_encoded[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        known_categories = set(self.label_encoders[feature].classes_)
                        df_encoded[feature] = df[feature].astype(str).apply(
                            lambda x: x if x in known_categories else 'Unknown'
                        )
                        # Add 'Unknown' to encoder if not present
                        if 'Unknown' not in known_categories:
                            classes = list(self.label_encoders[feature].classes_) + ['Unknown']
                            self.label_encoders[feature].classes_ = np.array(classes)
                        df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature])
                    else:
                        df_encoded[feature] = 0  # Default encoding
                        
        return df_encoded

class CloudVsLocalRecommender:
    """Recommends whether to train locally or on cloud"""
    
    def __init__(self):
        # Cloud pricing (approximate USD per hour)
        self.cloud_pricing = {
            'A100': 3.0,
            'V100': 2.0, 
            'H100': 5.0,
            'RTX 3090': 1.5,
            'RTX 4090': 2.5,
            'T4': 0.5
        }
        
        # Model complexity thresholds
        self.complexity_thresholds = {
            'small': 50e6,      # < 50M parameters
            'medium': 500e6,    # 50M - 500M parameters  
            'large': 10e9,      # 500M - 10B parameters
            'xlarge': float('inf') # > 10B parameters
        }
        
    def analyze_feasibility(self, 
                           model_params: int,
                           memory_requirement: float,
                           local_hardware: HardwareSpec) -> Dict[str, float]:
        """Analyze if local training is feasible"""
        scores = {}
        
        # Memory feasibility (GPU memory)
        memory_ratio = memory_requirement / local_hardware.gpu_memory_gb
        if memory_ratio <= 0.8:
            scores['memory'] = 1.0
        elif memory_ratio <= 1.0:
            scores['memory'] = 0.7
        else:
            scores['memory'] = max(0.0, 1.0 - (memory_ratio - 1.0))
            
        # Model complexity feasibility
        if model_params < self.complexity_thresholds['small']:
            scores['complexity'] = 1.0
        elif model_params < self.complexity_thresholds['medium']:
            scores['complexity'] = 0.8
        elif model_params < self.complexity_thresholds['large']:
            scores['complexity'] = 0.5
        else:
            scores['complexity'] = 0.2
            
        # Hardware capability score
        gpu_capability = {
            'RTX 3060': 0.3, 'RTX 3070': 0.5, 'RTX 3080': 0.7,
            'RTX 3090': 0.8, 'RTX 4070': 0.6, 'RTX 4080': 0.8,
            'RTX 4090': 0.9, 'A100': 1.0, 'H100': 1.0, 'V100': 0.7
        }
        scores['hardware'] = gpu_capability.get(local_hardware.gpu_type, 0.5)
        
        return scores
    
    def recommend_training_location(self,
                                  model_params: int,
                                  flops: float,
                                  memory_requirement: float,
                                  local_hardware: HardwareSpec,
                                  local_prediction: Dict[str, float],
                                  cloud_prediction: Dict[str, float]) -> TrainingRecommendation:
        """Generate comprehensive training recommendation"""
        
        # Analyze feasibility
        feasibility_scores = self.analyze_feasibility(model_params, memory_requirement, local_hardware)
        overall_feasibility = np.mean(list(feasibility_scores.values()))
        
        reasoning = []
        
        # Memory analysis
        if feasibility_scores['memory'] < 0.5:
            reasoning.append(f"⚠️  Insufficient GPU memory ({local_hardware.gpu_memory_gb}GB) for model requirements ({memory_requirement:.1f}GB)")
        elif feasibility_scores['memory'] < 0.8:
            reasoning.append(f"⚠️  Tight GPU memory constraints - may require optimization")
        else:
            reasoning.append(f"✅ Sufficient GPU memory available")
            
        # Complexity analysis
        if model_params > self.complexity_thresholds['large']:
            reasoning.append(f"⚠️  Very large model ({model_params/1e9:.1f}B parameters) - cloud training recommended")
        elif model_params > self.complexity_thresholds['medium']:
            reasoning.append(f"⚠️  Large model ({model_params/1e6:.0f}M parameters) - consider cloud for faster training")
        else:
            reasoning.append(f"✅ Model size ({model_params/1e6:.0f}M parameters) suitable for local training")
            
        # Cost comparison
        local_energy_cost = local_prediction['energy_usage'] * local_hardware.power_cost_per_kwh
        total_local_cost = local_energy_cost
        cloud_cost = cloud_prediction['cost_usd']
        
        cost_savings = cloud_cost - total_local_cost
        
        if cost_savings > 10:
            reasoning.append(f"💰 Significant cost savings with local training (${cost_savings:.2f})")
        elif cost_savings > 0:
            reasoning.append(f"💰 Moderate cost savings with local training (${cost_savings:.2f})")
        else:
            reasoning.append(f"💸 Cloud training more cost effective (${abs(cost_savings):.2f} savings)")
            
        # Time comparison
        time_difference = local_prediction['training_time'] - cloud_prediction['training_time']
        
        if time_difference > 4:
            reasoning.append(f"⏱️  Cloud training significantly faster ({time_difference:.1f}h time savings)")
        elif time_difference > 0:
            reasoning.append(f"⏱️  Cloud training somewhat faster ({time_difference:.1f}h time savings)")
        else:
            reasoning.append(f"⏱️  Similar training times")
            
        # Hardware capability
        if feasibility_scores['hardware'] < 0.5:
            reasoning.append(f"⚠️  Local GPU ({local_hardware.gpu_type}) may be underpowered for this model")
        elif feasibility_scores['hardware'] < 0.8:
            reasoning.append(f"⚠️  Local GPU adequate but not optimal")
        else:
            reasoning.append(f"✅ Local GPU well-suited for this model")
            
        # Make recommendation
        local_score = (
            feasibility_scores['memory'] * 0.3 +
            feasibility_scores['complexity'] * 0.2 + 
            feasibility_scores['hardware'] * 0.2 +
            (1 if cost_savings > 0 else 0) * 0.2 +
            (1 if time_difference <= 2 else 0) * 0.1
        )
        
        # Final decision logic
        if overall_feasibility < 0.4:
            recommendation = "cloud"
            confidence = 0.9
            reasoning.insert(0, "🌟 RECOMMENDATION: CLOUD - Local training not feasible")
        elif local_score > 0.7 and cost_savings > 5:
            recommendation = "local"
            confidence = 0.8
            reasoning.insert(0, "🌟 RECOMMENDATION: LOCAL - Good feasibility and cost savings")
        elif time_difference > 6 and cloud_cost < total_local_cost * 2:
            recommendation = "cloud"
            confidence = 0.7
            reasoning.insert(0, "🌟 RECOMMENDATION: CLOUD - Much faster with reasonable cost")
        elif overall_feasibility > 0.6:
            recommendation = "local" 
            confidence = 0.6
            reasoning.insert(0, "🌟 RECOMMENDATION: LOCAL - Feasible with acceptable trade-offs")
        else:
            recommendation = "cloud"
            confidence = 0.6
            reasoning.insert(0, "🌟 RECOMMENDATION: CLOUD - Better overall option")
        
        return TrainingRecommendation(
            recommendation=recommendation,
            confidence=confidence,
            local_prediction=local_prediction,
            cloud_prediction=cloud_prediction,
            reasoning=reasoning,
            cost_savings=cost_savings,
            time_difference=time_difference,
            feasibility_score=overall_feasibility
        )

class MLPerformancePredictor:
    """Main ML performance prediction system with local vs cloud recommendations"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.scraper = BenchmarkScraper()
        self.feature_engineer = FeatureEngineer()
        self.recommender = CloudVsLocalRecommender()
        
        # Models for different targets
        self.time_model = None
        self.energy_model = None
        self.cost_model = None
        
        # Model metadata
        self.model_metadata = {
            'last_updated': None,
            'num_samples': 0,
            'feature_columns': []
        }
        
    def collect_data(self, 
                    scrape_web: bool = True,
                    csv_files: List[str] = None,
                    json_files: List[str] = None) -> List[BenchmarkData]:
        """Collect benchmark data from all sources"""
        all_benchmarks = []
        
        # Web scraping (synthetic data)
        if scrape_web:
            logger.info("Generating synthetic benchmark data...")
            all_benchmarks.extend(self.scraper.scrape_mlperf_data())
            
        # CSV files
        if csv_files:
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    logger.info(f"Ingesting CSV: {csv_file}")
                    all_benchmarks.extend(DataIngestion.ingest_csv(csv_file))
                else:
                    logger.warning(f"CSV file not found: {csv_file}")
                
        logger.info(f"Collected {len(all_benchmarks)} benchmark data points")
        return all_benchmarks
    
    def train_models(self, benchmarks: List[BenchmarkData], model_type: str = "xgboost"):
        """Train ML models to predict performance metrics"""
        logger.info("Training performance prediction models...")
        
        # Extract and encode features
        df = self.feature_engineer.extract_features(benchmarks)
        df_encoded = self.feature_engineer.encode_features(df, fit=True)
        
        # Prepare features and targets
        target_columns = ['training_time', 'energy_usage', 'cost_usd']
        feature_columns = [col for col in df_encoded.columns if col not in target_columns]
        
        X = df_encoded[feature_columns]
        
        # Store feature columns for later prediction
        self.model_metadata['feature_columns'] = feature_columns
        
        # Train separate models for each target
        models = {}
        
        for target in target_columns:
            y = df_encoded[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            if model_type == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            else:  # random_forest
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{target} model - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            models[target] = model
            
        # Store models
        self.time_model = models['training_time']
        self.energy_model = models['energy_usage']
        self.cost_model = models['cost_usd']
        
        # Update metadata
        self.model_metadata.update({
            'last_updated': datetime.now(),
            'num_samples': len(benchmarks),
            'model_type': model_type
        })
        
        logger.info("Model training completed successfully!")
    
    def predict_performance(self, 
                          model_name: str,
                          dataset: str,
                          gpu_type: str,
                          cpu_type: str = "Intel Xeon",
                          batch_size: int = 32,
                          epochs: int = 100,
                          dataset_size: int = 100000,
                          model_params: int = 50000000,
                          flops: float = 1e10,
                          memory_gb: float = 16,
                          framework: str = "PyTorch",
                          precision: str = "FP32") -> Dict[str, float]:
        """Predict performance metrics using trained models"""
        
        if not self.models_loaded():
            logger.error("No trained models available. Using fallback predictions.")
            return self._fallback_prediction(model_params, flops, gpu_type, epochs)
        
        # Create feature vector
        feature_dict = {
            'model_params': model_params,
            'flops': flops,
            'dataset_size': dataset_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'memory_gb': memory_gb,
            'model_name': model_name,
            'dataset': dataset,
            'gpu_type': gpu_type,
            'cpu_type': cpu_type,
            'framework': framework,
            'precision': precision,
            'flops_per_param': flops / max(model_params, 1),
            'params_per_batch': model_params / max(batch_size, 1),
            'dataset_batches': dataset_size / max(batch_size, 1),
            'compute_intensity': flops / max(memory_gb, 1),
            'memory_per_param': memory_gb * 1e9 / max(model_params, 1),
            'model_params_log': np.log1p(model_params),
            'flops_log': np.log1p(flops),
            'dataset_size_log': np.log1p(dataset_size),
            'flops_per_param_log': np.log1p(flops / max(model_params, 1))
        }
        
        # Create DataFrame and encode
        df = pd.DataFrame([feature_dict])
        df_encoded = self.feature_engineer.encode_features(df, fit=False)
        
        # Select only the features used during training
        X = df_encoded[self.model_metadata['feature_columns']]
        
        # Make predictions
        predictions = {}
        
        try:
            predictions['training_time'] = max(0.1, float(self.time_model.predict(X)[0]))
            predictions['energy_usage'] = max(0.01, float(self.energy_model.predict(X)[0]))
            predictions['cost_usd'] = max(0.1, float(self.cost_model.predict(X)[0]))
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback_prediction(model_params, flops, gpu_type, epochs)
        
        return predictions
    
    def recommend_training_approach(self,
                                  model_name: str,
                                  dataset: str = "Custom Dataset",
                                  batch_size: int = 32,
                                  epochs: int = 100,
                                  dataset_size: int = 100000,
                                  model_params: int = 50000000,
                                  flops: float = 1e10,
                                  framework: str = "PyTorch",
                                  precision: str = "FP32",
                                  local_hardware: HardwareSpec = None) -> TrainingRecommendation:
        """Get recommendation for training location (local vs cloud)"""
        
        # Default local hardware if not provided
        if local_hardware is None:
            local_hardware = HardwareSpec(
                gpu_type="RTX 3090",
                gpu_memory_gb=24,
                cpu_type="Intel Core i9",
                ram_gb=64,
                storage_gb=1000,
                power_cost_per_kwh=0.12
            )
        
        # Estimate memory requirement (rough approximation)
        memory_requirement = self._estimate_memory_requirement(model_params, batch_size, precision)
        
        # Get predictions for local hardware
        local_prediction = self.predict_performance(
            model_name=model_name,
            dataset=dataset,
            gpu_type=local_hardware.gpu_type,
            cpu_type=local_hardware.cpu_type,
            batch_size=batch_size,
            epochs=epochs,
            dataset_size=dataset_size,
            model_params=model_params,
            flops=flops,
            memory_gb=local_hardware.gpu_memory_gb,
            framework=framework,
            precision=precision
        )
        
        # Get predictions for optimal cloud hardware (A100 by default)
        cloud_prediction = self.predict_performance(
            model_name=model_name,
            dataset=dataset,
            gpu_type="A100",
            cpu_type="Intel Xeon",
            batch_size=batch_size,
            epochs=epochs,
            dataset_size=dataset_size,
            model_params=model_params,
            flops=flops,
            memory_gb=40,  # A100 memory
            framework=framework,
            precision=precision
        )
        
        # Get recommendation
        recommendation = self.recommender.recommend_training_location(
            model_params=model_params,
            flops=flops,
            memory_requirement=memory_requirement,
            local_hardware=local_hardware,
            local_prediction=local_prediction,
            cloud_prediction=cloud_prediction
        )
        
        return recommendation
    
    def _estimate_memory_requirement(self, model_params: int, batch_size: int, precision: str) -> float:
        """Estimate GPU memory requirement in GB"""
        # Rough estimation based on model parameters and batch size
        bytes_per_param = 4 if precision == "FP32" else 2  # FP16 uses half the memory
        
        # Model weights + gradients + optimizer states (Adam = 2x params) + activations
        memory_bytes = model_params * bytes_per_param * (1 + 1 + 2) + (batch_size * model_params * bytes_per_param * 0.1)
        
        memory_gb = memory_bytes / (1024**3)
        return max(1.0, memory_gb)  # Minimum 1GB
    
    def models_loaded(self) -> bool:
        """Check if models are trained and loaded"""
        return (self.time_model is not None and 
                self.energy_model is not None and 
                self.cost_model is not None)
    
    def _fallback_prediction(self, model_params: int, flops: float, gpu_type: str, epochs: int) -> Dict[str, float]:
        """Fallback prediction using simple heuristics"""
        logger.info("Using fallback prediction method")
        
        # Simple heuristic based on model complexity
        complexity_factor = (model_params / 1e8) * (flops / 1e12)
        
        # GPU performance multipliers
        gpu_multipliers = {
            'H100': 0.5, 'A100': 0.7, 'V100': 1.0, 
            'RTX 4090': 0.8, 'RTX 3090': 1.2, 'RTX 3080': 1.5
        }
        gpu_mult = gpu_multipliers.get(gpu_type, 1.0)
        
        base_time = complexity_factor * gpu_mult * epochs / 100
        training_time = max(0.5, base_time)
        
        # Estimate energy (assuming 400W average power)
        energy_usage = training_time * 0.4
        
        # Estimate cost (assuming $3/hour average)
        cost_usd = training_time * 3.0
        
        return {
            'training_time': training_time,
            'energy_usage': energy_usage,
            'cost_usd': cost_usd
        }

def print_recommendation_report(recommendation: TrainingRecommendation, model_name: str):
    """Print a detailed recommendation report"""
    print(f"\n{'='*60}")
    print(f"🤖 TRAINING RECOMMENDATION FOR: {model_name}")
    print(f"{'='*60}")
    
    # Main recommendation
    rec_emoji = "🏠" if recommendation.recommendation == "local" else "☁️"
    print(f"\n{rec_emoji} RECOMMENDATION: {recommendation.recommendation.upper()}")
    print(f"🎯 Confidence: {recommendation.confidence*100:.1f}%")
    print(f"📊 Feasibility Score: {recommendation.feasibility_score*100:.1f}%")
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<20} {'Local':<15} {'Cloud':<15} {'Difference':<15}")
    print(f"{'-'*65}")
    
    local = recommendation.local_prediction
    cloud = recommendation.cloud_prediction
    
    time_diff = local['training_time'] - cloud['training_time']
    time_diff_str = f"{time_diff:+.1f}h"
    
    energy_diff = local['energy_usage'] - cloud['energy_usage']  
    energy_diff_str = f"{energy_diff:+.1f} kWh"
    
    cost_diff = local['cost_usd'] - cloud['cost_usd']
    cost_diff_str = f"${cost_diff:+.2f}"
    
    print(f"{'Training Time':<20} {local['training_time']:<14.1f}h {cloud['training_time']:<14.1f}h {time_diff_str:<15}")
    print(f"{'Energy Usage':<20} {local['energy_usage']:<14.1f} {cloud['energy_usage']:<14.1f} {energy_diff_str:<15}")
    print(f"{'Cost (USD)':<20} ${local['cost_usd']:<13.2f} ${cloud['cost_usd']:<13.2f} {cost_diff_str:<15}")
    
    # Cost savings highlight
    if recommendation.cost_savings > 0:
        print(f"\n💰 POTENTIAL SAVINGS: ${recommendation.cost_savings:.2f} with local training")
    else:
        print(f"\n💸 ADDITIONAL COST: ${abs(recommendation.cost_savings):.2f} for local training")
    
    # Detailed reasoning
    print(f"\n🔍 DETAILED ANALYSIS:")
    for reason in recommendation.reasoning:
        print(f"   {reason}")
    
    print(f"\n{'='*60}")

def main():
    """Main function to run the enhanced performance predictor"""
    print("🚀 Starting Enhanced ML Performance Predictor with Local vs Cloud Recommendations...")
    
    # Initialize the predictor
    predictor = MLPerformancePredictor()
    
    # Collect data
    print("\n📊 Collecting benchmark data...")
    
    # Try CSV first, fallback to synthetic
    csv_file = 'dataset.csv'
    if os.path.exists(csv_file):
        print(f"✅ Found {csv_file}, loading real benchmark data...")
        benchmarks = predictor.collect_data(
            scrape_web=True,  # Also include synthetic data
            csv_files=[csv_file]
        )
    else:
        print(f"⚠️  {csv_file} not found, using synthetic data only...")
        benchmarks = predictor.collect_data(scrape_web=True)
    
    print(f"✅ Collected {len(benchmarks)} benchmark records")
    
    # Train models
    print("\n🧠 Training ML models...")
    try:
        predictor.train_models(benchmarks, model_type="xgboost")
        print("✅ Models trained successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Define user's local hardware
    user_local_hardware = HardwareSpec(
        gpu_type="RTX 3090",
        gpu_memory_gb=24,
        cpu_type="Intel Core i9",
        ram_gb=32,
        storage_gb=1000,
        power_cost_per_kwh=0.12  # USD per kWh
    )
    
    print(f"\n💻 User's Local Hardware Configuration:")
    print(f"   GPU: {user_local_hardware.gpu_type} ({user_local_hardware.gpu_memory_gb}GB)")
    print(f"   CPU: {user_local_hardware.cpu_type}")
    print(f"   RAM: {user_local_hardware.ram_gb}GB")
    print(f"   Power Cost: ${user_local_hardware.power_cost_per_kwh}/kWh")
    
    # Test cases for recommendations
    print("\n🔮 Generating Training Recommendations...")
    
    '''test_models = [
        {
            "model_name": "ResNet-50",
            "dataset": "ImageNet",
            "model_params": 25600000,
            "flops": 4.1e9,
            "batch_size": 32,
            "epochs": 90
        },
        {
            "model_name": "BERT-Large",
            "dataset": "Wikipedia", 
            "model_params": 340000000,
            "flops": 1.3e11,
            "batch_size": 16,
            "epochs": 3
        },
        {
            "model_name": "GPT-3-Small",
            "dataset": "OpenWebText",
            "model_params": 6700000000,  # 6.7B parameters
            "flops": 5e12,
            "batch_size": 8,
            "epochs": 1
        },
        {
            "model_name": "Custom-CNN",
            "dataset": "Custom Dataset",
            "model_params": 15000000,
            "flops": 2e10,
            "batch_size": 64,
            "epochs": 200
        }
    ]
    
    for i, model_config in enumerate(test_models, 1):
        print(f"\n🧪 Test Case {i}: {model_config['model_name']}")
        try:
            recommendation = predictor.recommend_training_approach(
                **model_config,
                local_hardware=user_local_hardware
            )
            
            # Print detailed recommendation report
            print_recommendation_report(recommendation, model_config['model_name'])
            
        except Exception as e:
            print(f"❌ Recommendation failed for {model_config['model_name']}: {e}")
    '''
    # Interactive mode
    print(f"\n🎯 INTERACTIVE MODE")
    print("Enter your model details to get a personalized recommendation:")
    
    try:
        model_name = input("Model name (e.g., 'My-Custom-Model'): ") or "Custom-Model"
        model_params = int(input("Number of parameters (e.g., 50000000): ") or "50000000")
        epochs = int(input("Number of epochs (e.g., 100): ") or "100")
        batch_size = int(input("Batch size (e.g., 32): ") or "32")
        
        print(f"\n🔄 Analyzing {model_name}...")
        
        recommendation = predictor.recommend_training_approach(
            model_name=model_name,
            model_params=model_params,
            epochs=epochs,
            batch_size=batch_size,
            local_hardware=user_local_hardware
        )
        
        print_recommendation_report(recommendation, model_name)
        
    except KeyboardInterrupt:
        print("\n👋 Interactive mode cancelled by user")
    except Exception as e:
        print(f"❌ Interactive recommendation failed: {e}")
    
    print(f"\n🎉 Enhanced ML Performance Predictor completed!")
    print("\n💡 Usage Summary:")
    print("   • The system analyzes your model requirements")
    print("   • Compares local vs cloud training options")  
    print("   • Considers cost, time, and feasibility factors")
    print("   • Provides detailed recommendations with reasoning")
    print("\n📚 Key Features:")
    print("   • Memory feasibility analysis")
    print("   • Hardware capability assessment") 
    print("   • Cost-benefit comparison")
    print("   • Performance prediction for both options")
    print("   • Confidence scoring and detailed reasoning")

if __name__ == "__main__":
    main()