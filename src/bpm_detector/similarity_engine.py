"""Music similarity and feature extraction module."""

from typing import Any, Dict, List, Optional

import numpy as np


class SimilarityEngine:
    """Engine for comparing audio tracks and determining similarity."""

    def __init__(self):
        """Initialize similarity engine."""
        # Define weights for different feature categories in the 'weighted' method
        self.weights = {
            'bpm': 1.5,
            'key': 1.0,
            'energy': 1.2,
            'complexity': 0.8,
            'brightness': 1.0,
        }

    def extract_feature_vector(self, analysis_results: Dict[str, Any]) -> List[float]:
        """Extract a numerical feature vector from analysis results.
        
        Args:
            analysis_results: Dictionary from AudioAnalyzer.analyze_file()
            
        Returns:
            List of numerical features
        """
        vector = []
        
        # 1. BPM Feature (Normalized assuming 60-200 BPM range)
        basic_info = analysis_results.get('basic_info', {})
        bpm = basic_info.get('bpm', 120.0)
        bpm_norm = max(0.0, min(1.0, (bpm - 60.0) / 140.0))
        vector.append(bpm_norm)
        
        # 2. Structure/Complexity
        structure = analysis_results.get('structure', {})
        complexity = structure.get('structural_complexity', 0.5)
        vector.append(complexity)
        
        # 3. Rhythm/Syncopation
        rhythm = analysis_results.get('rhythm', {})
        syncopation = rhythm.get('syncopation_level', 0.0)
        vector.append(syncopation)
        
        # 4. Energy/Dynamics (Placeholder for actual dynamics analyzer integration)
        dynamics = analysis_results.get('dynamics', {})
        energy = dynamics.get('overall_energy', 0.5)
        vector.append(energy)
        
        # 5. Timbre/Brightness (Placeholder for actual timbre analyzer)
        timbre = analysis_results.get('timbre', {})
        brightness = timbre.get('brightness', 0.5)
        vector.append(brightness)
        
        return vector

    def generate_similarity_matrix(self, feature_vectors: List[List[float]], method: str = 'weighted') -> np.ndarray:
        """Generate a similarity matrix for a list of feature vectors.
        
        Args:
            feature_vectors: List of feature vectors
            method: Similarity calculation method ('euclidean', 'cosine', 'weighted')
            
        Returns:
            2D numpy array representing the similarity matrix (0.0 to 1.0)
        """
        n = len(feature_vectors)
        matrix = np.zeros((n, n))
        
        if n == 0:
            return matrix
            
        vectors_np = np.array(feature_vectors)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    sim = self._calculate_similarity(vectors_np[i], vectors_np[j], method)
                    matrix[i, j] = sim
                    matrix[j, i] = sim
                    
        return matrix

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray, method: str) -> float:
        """Calculate similarity between two vectors."""
        # Ensure same length
        min_len = min(len(vec1), len(vec2))
        v1 = vec1[:min_len]
        v2 = vec2[:min_len]
        
        if method == 'cosine':
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
            
        elif method == 'euclidean':
            dist = np.linalg.norm(v1 - v2)
            # Convert distance to similarity (0 to 1)
            return float(1.0 / (1.0 + dist))
            
        elif method == 'weighted':
            # Simple weighted absolute difference
            diff = np.abs(v1 - v2)
            weights_list = list(self.weights.values())[:min_len]
            # Pad weights if necessary
            if len(weights_list) < min_len:
                weights_list.extend([1.0] * (min_len - len(weights_list)))
            
            weights_np = np.array(weights_list)
            weighted_diff = np.sum(diff * weights_np)
            max_diff = np.sum(weights_np) # Assuming max difference per feature is 1.0
            
            if max_diff == 0:
                return 0.0
                
            sim = 1.0 - (weighted_diff / max_diff)
            return float(max(0.0, min(1.0, sim)))
            
        else:
            # Default to basic euclidean similarity
            return float(1.0 / (1.0 + np.linalg.norm(v1 - v2)))