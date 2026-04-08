"""Entry point for building the Phase 1 persona dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from app.config import (
    DATASET_PATH,
    DEFAULT_AI_POST_COUNT,
    DEFAULT_TWEET_COUNT,
    PREPROCESSED_DATASET_PATH,
    PROCESSED_DATA_DIR,
)
from app.data_acquisition.dataset_builder import build_dataset
from app.feature_extraction import FeatureExtractionConfig, build_feature_matrix
from app.attribution_clustering import (
    AttributionClusteringConfig,
    run_attribution_clustering_pipeline,
)
from app.persona_detection import (
    TrainingConfig,
    load_model,
    predict_usernames_from_feature_frame,
    save_model,
    train_model,
    visualize_feature_importance,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset generation."""

    parser = argparse.ArgumentParser(
        description="Build a labeled human/AI persona dataset for a target account."
    )
    parser.add_argument(
        "username",
        nargs="?",
        default="sample_user",
        help="Target account username to scrape (default: sample_user).",
    )
    parser.add_argument(
        "--max-tweets",
        type=int,
        default=DEFAULT_TWEET_COUNT,
        help=f"Maximum human tweets to scrape (default: {DEFAULT_TWEET_COUNT}).",
    )
    parser.add_argument(
        "--ai-posts",
        type=int,
        default=DEFAULT_AI_POST_COUNT,
        help=f"Number of AI posts to generate (default: {DEFAULT_AI_POST_COUNT}).",
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=400,
        help="Maximum number of TF-IDF features for Phase 3 extraction (default: 400).",
    )
    parser.add_argument(
        "--no-feature-export",
        action="store_true",
        help="Disable CSV export of Phase 3 feature outputs.",
    )
    parser.add_argument(
        "--train-phase4",
        action="store_true",
        help="Train Phase 4 classifiers and select the best by F1 score.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable GridSearchCV hyperparameter tuning for each model.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for Phase 4 training (default: 5).",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Top N important features to show for RF/XGBoost (default: 20).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROCESSED_DATA_DIR / "best_persona_model.pkl",
        help="Output path for saving the best Phase 4 model.",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Generate confusion matrix and feature importance visualizations.",
    )
    parser.add_argument(
        "--interactive-viz",
        action="store_true",
        help="Prefer interactive Plotly visualizations when available.",
    )
    parser.add_argument(
        "--predict-features-csv",
        type=Path,
        default=None,
        help="CSV containing username + Phase 3 features for real-time prediction.",
    )
    parser.add_argument(
        "--predict-model",
        type=Path,
        default=PROCESSED_DATA_DIR / "best_persona_model.pkl",
        help="Model bundle (.pkl) to load for prediction mode.",
    )
    parser.add_argument(
        "--run-phase5",
        action="store_true",
        help="Run Phase 5 attribution and clustering after prediction mode.",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=0.5,
        help="DBSCAN eps for Phase 5 clustering (default: 0.5).",
    )
    parser.add_argument(
        "--cluster-min-samples",
        type=int,
        default=3,
        help="DBSCAN min_samples for Phase 5 clustering (default: 3).",
    )
    parser.add_argument(
        "--cluster-edge-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for graph edges in Phase 5 (default: 0.7).",
    )
    parser.add_argument(
        "--cluster-graph-output",
        type=Path,
        default=None,
        help="Optional output path for Phase 5 graph visualization (.png or .html).",
    )
    parser.add_argument(
        "--cluster-export-json",
        type=Path,
        default=None,
        help="Optional output path to export Phase 5 graph JSON.",
    )
    parser.add_argument(
        "--cluster-export-csv",
        type=Path,
        default=None,
        help="Optional output path to export cluster assignments CSV.",
    )
    return parser.parse_args()


def run_prediction_mode(args: argparse.Namespace) -> int:
    """Predict persona labels from a feature CSV using a saved model."""

    if args.predict_features_csv is None:
        return 0

    if not args.predict_features_csv.exists():
        print(f"Prediction CSV not found: {args.predict_features_csv}")
        return 1

    if not args.predict_model.exists():
        print(f"Model file not found: {args.predict_model}")
        return 1

    try:
        model_bundle = load_model(args.predict_model)
        feature_frame = pd.read_csv(args.predict_features_csv)
        predictions = predict_usernames_from_feature_frame(model_bundle, feature_frame)
    except Exception as exc:  # pragma: no cover - top-level execution path
        print(f"Failed to run prediction mode: {exc}")
        return 1

    print("\nPrediction output:")
    print(predictions.head(20))

    if args.run_phase5:
        if "username" not in feature_frame.columns:
            print("Phase 5 requires a 'username' column in the prediction feature CSV.")
            return 1

        X_phase5 = feature_frame.drop(columns=["username"])
        phase5_config = AttributionClusteringConfig(
            eps=args.cluster_eps,
            min_samples=args.cluster_min_samples,
            edge_threshold=args.cluster_edge_threshold,
        )

        try:
            phase5_result = run_attribution_clustering_pipeline(
                X=X_phase5,
                predictions=predictions,
                config=phase5_config,
                graph_output_path=args.cluster_graph_output,
                graph_interactive=args.interactive_viz,
                save_clusters_csv_path=args.cluster_export_csv,
                export_graph_json_path=args.cluster_export_json,
                add_louvain_communities=True,
            )
        except Exception as exc:  # pragma: no cover - top-level execution path
            print(f"Failed during Phase 5 attribution/clustering: {exc}")
            return 1

        clusters = phase5_result["clusters"]
        isolated = phase5_result["isolated_personas"]
        print("\nPhase 5 attribution summary:")
        print(f"- AI personas clustered: {len(phase5_result['cluster_assignments'])}")
        print(f"- Cluster groups found: {len([cid for cid in clusters.keys() if int(cid) != -1])}")
        print(f"- Isolated personas: {len(isolated)}")

        if args.cluster_export_csv is not None:
            print(f"- Cluster CSV exported: {args.cluster_export_csv}")
        if args.cluster_export_json is not None:
            print(f"- Graph JSON exported: {args.cluster_export_json}")
        if args.cluster_graph_output is not None:
            print(f"- Graph visualization saved: {args.cluster_graph_output}")

    return 0


def main() -> int:
    """Run Phase 1-4 pipeline and print output summaries."""

    args = parse_args()
    if args.predict_features_csv is not None:
        return run_prediction_mode(args)

    try:
        dataset = build_dataset(
            username=args.username,
            max_human_tweets=args.max_tweets,
            num_ai_posts=args.ai_posts,
        )
    except Exception as exc:  # pragma: no cover - top-level execution path
        print(f"Failed to build dataset: {exc}")
        return 1

    print(dataset.head())
    print(f"\nDataset saved to: {DATASET_PATH}")
    print(f"File exists: {DATASET_PATH.exists()}")
    print(f"Preprocessed dataset saved to: {PREPROCESSED_DATASET_PATH}")
    print(f"File exists: {PREPROCESSED_DATASET_PATH.exists()}")

    try:
        preprocessed_df = pd.read_csv(PREPROCESSED_DATASET_PATH)
        config = FeatureExtractionConfig(tfidf_max_features=args.tfidf_max_features)
        X, y = build_feature_matrix(
            dataset=preprocessed_df,
            config=config,
            save_debug_csv=not args.no_feature_export,
            debug_output_dir=PROCESSED_DATA_DIR,
        )
    except Exception as exc:  # pragma: no cover - top-level execution path
        print(f"Failed to build Phase 3 feature matrix: {exc}")
        return 1

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    if not X.empty:
        print("Sample feature columns:")
        print(X.columns[:20].tolist())
    if not args.no_feature_export:
        print(f"Phase 3 CSV outputs saved under: {PROCESSED_DATA_DIR}")

    if args.train_phase4:
        try:
            training_config = TrainingConfig(
                cv_folds=args.cv_folds,
                use_grid_search=args.grid_search,
                top_n_features=args.top_features,
            )
            training_result = train_model(
                X=X,
                y=y,
                config=training_config,
                plot_confusion=args.plot_metrics,
                output_dir=PROCESSED_DATA_DIR,
                interactive=args.interactive_viz,
            )
        except Exception as exc:  # pragma: no cover - top-level execution path
            print(f"Failed during Phase 4 training: {exc}")
            return 1

        best_name = training_result["best_model_name"]
        best_metrics = training_result["metrics"][best_name]
        print("\nPhase 4 model comparison:")
        for model_name, metrics in training_result["metrics"].items():
            print(
                f"- {model_name}: "
                f"acc={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}, "
                f"cv_f1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}"
            )

        print(f"\nBest model: {best_name} (F1={best_metrics['f1']:.4f})")
        model_path = save_model(
            model=training_result["best_model"],
            file_path=args.model_output,
            feature_names=training_result["feature_names"],
            model_name=best_name,
        )
        print(f"Saved best model to: {model_path}")

        rf_importance = training_result["feature_importance"].get("RandomForest", pd.DataFrame())
        xgb_importance = training_result["feature_importance"].get("XGBoost", pd.DataFrame())

        if not rf_importance.empty:
            print(f"\nTop {args.top_features} RandomForest features:")
            print(rf_importance.head(args.top_features))
            if args.plot_metrics:
                visualize_feature_importance(
                    rf_importance,
                    model_name="RandomForest",
                    output_dir=PROCESSED_DATA_DIR,
                    interactive=args.interactive_viz,
                )

        if not xgb_importance.empty:
            print(f"\nTop {args.top_features} XGBoost features:")
            print(xgb_importance.head(args.top_features))
            if args.plot_metrics:
                visualize_feature_importance(
                    xgb_importance,
                    model_name="XGBoost",
                    output_dir=PROCESSED_DATA_DIR,
                    interactive=args.interactive_viz,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
