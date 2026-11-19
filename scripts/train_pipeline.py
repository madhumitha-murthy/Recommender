"""
End-to-end training pipeline.

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --optuna-trials 20  # quick run
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.sql_features import build_customer_features, build_campaign_dataset, get_engine
from src.models.recommender import train_recommender, build_interaction_matrix
from src.models.uplift import train_uplift_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Train recommendation + uplift models")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("nbp-recommender")

    engine = get_engine()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build features from SQL
    logger.info("=" * 60)
    logger.info("STEP 1: Building features from SQL")
    logger.info("=" * 60)

    customer_features = build_customer_features(engine)
    customer_features.to_parquet(DATA_DIR / "customer_features.parquet", index=False)
    logger.info(f"Saved customer features: {customer_features.shape}")

    # 2. Load product changes for recommender
    logger.info("=" * 60)
    logger.info("STEP 2: Loading product changes")
    logger.info("=" * 60)

    import pandas as pd
    from sqlalchemy import text
    with engine.connect() as conn:
        product_changes = pd.read_sql("SELECT * FROM product_changes", conn)
    logger.info(f"Product changes: {len(product_changes):,} events")

    # 3. Train recommender
    logger.info("=" * 60)
    logger.info("STEP 3: Training recommendation model")
    logger.info("=" * 60)

    als, ranker, rec_metrics = train_recommender(
        customer_features, product_changes, n_optuna_trials=args.optuna_trials
    )
    logger.info(f"Recommender NDCG@5: {rec_metrics['val']['ndcg@5']:.4f}")

    # 4. Build campaign dataset and train uplift model
    logger.info("=" * 60)
    logger.info("STEP 4: Training uplift model")
    logger.info("=" * 60)

    campaign_dataset = build_campaign_dataset(engine)
    campaign_dataset.to_parquet(DATA_DIR / "campaign_dataset.parquet", index=False)

    uplift_model, uplift_metrics = train_uplift_model(campaign_dataset)
    logger.info(f"Uplift AUUC: {uplift_metrics['auuc']:.4f}")

    # 5. Summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Recommender val NDCG@5: {rec_metrics['val']['ndcg@5']:.4f}")
    logger.info(f"Recommender val Precision@5: {rec_metrics['val']['precision@5']:.4f}")
    logger.info(f"Uplift AUUC: {uplift_metrics['auuc']:.4f}")
    logger.info(f"Uplift Qini: {uplift_metrics['qini']:.4f}")
    logger.info(f"Models saved to: {DATA_DIR.parent / 'models'}")


if __name__ == "__main__":
    main()
