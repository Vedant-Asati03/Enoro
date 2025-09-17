"""
Evaluation metrics and utilities for ML models.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class MetricsCalculator:
    """
    Calculate various metrics for ML model evaluation.

    Provides metrics for both recommendation systems and
    classification tasks.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def calculate_recommendation_metrics(
        self, true_items: List[str], recommended_items: List[str], k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate recommendation metrics.

        Args:
            true_items: List of items user actually interacted with
            recommended_items: List of recommended items
            k: Number of top recommendations to consider

        Returns:
            Dictionary of metrics
        """
        # Take top-k recommendations
        top_k_recs = recommended_items[:k]

        # Convert to sets for easier calculation
        true_set = set(true_items)
        rec_set = set(top_k_recs)

        # Calculate metrics
        intersection = len(true_set.intersection(rec_set))

        # Precision@K: fraction of recommended items that are relevant
        precision_at_k = intersection / len(rec_set) if rec_set else 0.0

        # Recall@K: fraction of relevant items that are recommended
        recall_at_k = intersection / len(true_set) if true_set else 0.0

        # F1@K: harmonic mean of precision and recall
        f1_at_k = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if (precision_at_k + recall_at_k) > 0
            else 0.0
        )

        # Hit Rate@K: whether any recommended item is relevant
        hit_rate_at_k = 1.0 if intersection > 0 else 0.0

        return {
            f"precision_at_{k}": precision_at_k,
            f"recall_at_{k}": recall_at_k,
            f"f1_at_{k}": f1_at_k,
            f"hit_rate_at_{k}": hit_rate_at_k,
            "num_relevant_recommended": intersection,
            "total_recommended": len(rec_set),
            "total_relevant": len(true_set),
        }

    def calculate_ranking_metrics(
        self, true_items: List[str], recommended_items: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ranking quality metrics.

        Args:
            true_items: List of relevant items
            recommended_items: List of recommended items (in order)

        Returns:
            Dictionary of ranking metrics
        """
        true_set = set(true_items)

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, item in enumerate(recommended_items):
            if item in true_set:
                mrr = 1.0 / (i + 1)  # 1-indexed rank
                break

        # Average Precision (AP)
        relevant_found = 0
        precision_sum = 0.0

        for i, item in enumerate(recommended_items):
            if item in true_set:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i

        average_precision = precision_sum / len(true_set) if true_set else 0.0

        # Normalized Discounted Cumulative Gain (NDCG)
        ndcg = self._calculate_ndcg(true_items, recommended_items)

        return {"mrr": mrr, "average_precision": average_precision, "ndcg": ndcg}

    def _calculate_ndcg(
        self, true_items: List[str], recommended_items: List[str], k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        Args:
            true_items: List of relevant items
            recommended_items: List of recommended items
            k: Number of items to consider

        Returns:
            NDCG score
        """
        true_set = set(true_items)

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in true_set:
                # Relevance = 1 for relevant items, 0 for others
                relevance = 1.0
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(true_items), k)):
            idcg += 1.0 / np.log2(i + 2)

        # NDCG = DCG / IDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return ndcg

    def calculate_diversity_metrics(
        self, recommended_items: List[str], item_features: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate diversity metrics for recommendations.

        Args:
            recommended_items: List of recommended items
            item_features: Dict mapping item_id to list of features/categories

        Returns:
            Dictionary of diversity metrics
        """
        if len(recommended_items) < 2:
            return {"diversity": 0.0, "coverage": 0.0}

        # Calculate intra-list diversity (average pairwise dissimilarity)
        total_pairs = 0
        dissimilar_pairs = 0

        for i in range(len(recommended_items)):
            for j in range(i + 1, len(recommended_items)):
                item1 = recommended_items[i]
                item2 = recommended_items[j]

                features1 = set(item_features.get(item1, []))
                features2 = set(item_features.get(item2, []))

                # Jaccard dissimilarity
                union = features1.union(features2)
                intersection = features1.intersection(features2)

                if union:
                    similarity = len(intersection) / len(union)
                    dissimilarity = 1.0 - similarity
                    dissimilar_pairs += dissimilarity

                total_pairs += 1

        diversity = dissimilar_pairs / total_pairs if total_pairs > 0 else 0.0

        # Calculate coverage (how many different categories are covered)
        all_categories = set()
        for item in recommended_items:
            all_categories.update(item_features.get(item, []))

        # Get total possible categories
        total_categories = set()
        for features in item_features.values():
            total_categories.update(features)

        coverage = (
            len(all_categories) / len(total_categories) if total_categories else 0.0
        )

        return {
            "diversity": diversity,
            "coverage": coverage,
            "unique_categories": len(all_categories),
        }

    def calculate_classification_metrics(
        self, y_true: List[int], y_pred: List[int], average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class metrics

        Returns:
            Dictionary of classification metrics
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

    def calculate_model_performance_summary(
        self, test_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate summary statistics across multiple test results.

        Args:
            test_results: List of individual test result dictionaries

        Returns:
            Summary statistics
        """
        if not test_results:
            return {}

        # Extract all metric names
        all_metrics = set()
        for result in test_results:
            all_metrics.update(result.keys())

        # Calculate mean and std for each metric
        summary = {}
        for metric in all_metrics:
            values = [result.get(metric, 0.0) for result in test_results]
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)

        return summary
