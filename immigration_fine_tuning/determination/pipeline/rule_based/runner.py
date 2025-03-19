import argparse
import json
import time
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import yaml
from datetime import datetime

from ngram_determination_extraction import NgramDeterminationExtractor
from basic_determination_extraction import BasicDeterminationExtractor
from sparse_explicit_extraction import SparseExplicitExtractor

@dataclass
class ExperimentConfig:
    """Configuration for a determination extraction experiment."""
    model_name: str
    model_params: Dict[str, Any]
    experiment_name: Optional[str] = None
    description: Optional[str] = None

class DeterminationExperimentRunner:
    """Runner for determination extraction experiments."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent.parent.parent.parent
        self.models = self._get_available_models()
        self.experiment_history = []
        
    def _get_available_models(self) -> Dict[str, str]:
        """Returns the list of available determination extraction models."""
        return {
            'ngram': 'NgramDeterminationExtractor',
            'basic': 'BasicDeterminationExtractor',
            'sparse': 'SparseExplicitExtractor'  # Add our new model
        }
    
    def _load_model(self, model_name: str, **model_params) -> Any:
        """Load a model by name with given parameters."""
        if model_name == 'ngram':
            model = NgramDeterminationExtractor(**model_params)
        elif model_name == 'basic':
            model = BasicDeterminationExtractor(**model_params)
        elif model_name == 'sparse':
            model = SparseExplicitExtractor(**model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return model
    
    def run_experiment(self, config: Union[ExperimentConfig, List[ExperimentConfig]], 
                      data_paths: Optional[Dict[str, Path]] = None,
                      output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run one or more experiments.
        
        Args:
            config: Single experiment config or list of configs
            data_paths: Optional dictionary with custom data paths
            output_dir: Optional custom output directory
        
        Returns:
            Dictionary with experiment results
        """
        # Convert single config to list
        configs = [config] if isinstance(config, ExperimentConfig) else config
        
        # Setup paths
        paths = self._setup_paths(data_paths, output_dir)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = paths['output'] / f"experiment_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Run each experiment
        for exp_config in configs:
            exp_name = exp_config.experiment_name or f"{exp_config.model_name}_{timestamp}"
            print(f"\nRunning experiment: {exp_name}")
            if exp_config.description:
                print(f"Description: {exp_config.description}")
            
            # Run the model
            results = self._run_single_model(
                exp_config.model_name,
                paths['train'],
                paths['test'],
                paths['validation'],
                experiment_dir,
                exp_config.model_params
            )
            
            # Add experiment metadata
            results['experiment'] = {
                'name': exp_name,
                'description': exp_config.description,
                'timestamp': timestamp,
                'config': exp_config.__dict__
            }
            
            # Save individual results
            result_path = experiment_dir / f"{exp_name}_results.json"
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            all_results[exp_name] = results
            self.experiment_history.append(results)
        
        # Save combined results if multiple experiments
        if len(configs) > 1:
            self._save_combined_results(all_results, experiment_dir)
        
        return all_results
    
    def _setup_paths(self, data_paths: Optional[Dict[str, Path]], 
                    output_dir: Optional[Path]) -> Dict[str, Path]:
        """Setup and validate all required paths."""
        paths = {
            'train': Path(data_paths['train']) if data_paths and 'train' in data_paths 
                    else self.base_dir / "data" / "merged" / "train_enriched.csv",
            'test': Path(data_paths['test']) if data_paths and 'test' in data_paths 
                    else self.base_dir / "data" / "merged" / "test_enriched.csv",
            'validation': Path(data_paths['validation']) if data_paths and 'validation' in data_paths 
                         else Path(__file__).parent / "preprocessed_validation_set.csv",
            'output': Path(output_dir) if output_dir else Path(__file__).parent / "experiments"
        }
        
        # Validate paths
        if not paths['train'].exists() or not paths['test'].exists():
            print(f"Warning: Train/test data not found at default paths")
            print("Trying fallback paths...")
            # Try windows-specific path (adjust as needed)
            paths['train'] = Path("C:/Users/shil6369/cultural-law-interpretations/immigration_fine_tuning/data/merged/train_enriched.csv")
            paths['test'] = Path("C:/Users/shil6369/cultural-law-interpretations/immigration_fine_tuning/data/merged/test_enriched.csv")
        
        # Create output directory
        paths['output'].mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def _run_single_model(self, model_name: str, train_path: Path, test_path: Path, 
                         validation_path: Path, output_dir: Path, 
                         model_params: Dict = None) -> Dict[str, Any]:
        """Run a single model experiment."""
        print(f"Running {model_name} model...")
        start_time = time.time()
        
        # Load the model
        model_params = model_params or {}
        model = self._load_model(model_name, **model_params)
        
        # Load training examples
        model.load_training_examples(str(train_path), str(test_path), use_chunking=True)
        
        # Run evaluation
        results, metrics, analysis_paths = self._run_evaluation(model, validation_path, output_dir)
        
        # Get performance stats
        perf_stats = model.get_performance_stats()
        
        # Prepare results
        output = {
            'model_name': model_name,
            'model_params': model_params,
            'metrics': metrics,
            'performance': perf_stats,
            'total_runtime': f"{time.time() - start_time:.2f} seconds",
            'case_results': [
                {
                    'case_id': r['case_id'],
                    'metrics': r['metrics'],
                    'extracted_count': r['metrics']['extracted_count'],
                    'expected_count': r['metrics']['expected_count']
                } for r in results
            ],
            'analysis_files': analysis_paths
        }
        
        # Print results
        print(f"\nPerformance Stats:")
        for k, v in perf_stats.items():
            print(f"  {k}: {v}")
        
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        return output
    
    def _run_evaluation(self, model, validation_file: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[str]]:
        """Run evaluation on the validation set."""
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from evaluation import run_evaluation as eval_func
            
            print("\nRunning evaluation on validation set...")
            return eval_func(
                model, 
                validation_file=validation_file,
                output_dir=output_dir,
                analyze_failures=True
            )
            
        except Exception as e:
            print(f"Error running evaluation: {e}")
            import traceback
            traceback.print_exc()
            return [], {}, []
    
    def _save_combined_results(self, all_results: Dict[str, Any], output_dir: Path):
        """Save combined results from multiple experiments."""
        combined_results = {
            'experiments': list(all_results.keys()),
            'metrics_by_experiment': {name: results['metrics'] 
                                    for name, results in all_results.items()},
            'performance_by_experiment': {name: results['performance'] 
                                        for name, results in all_results.items()},
            'total_runtime': sum(float(results['total_runtime'].split()[0]) 
                               for results in all_results.values())
        }
        
        # Calculate average metrics across experiments
        avg_metrics = {}
        for metric in ['precision', 'recall', 'f1', 'exact_match']:
            values = [results['metrics'].get(metric, 0) for results in all_results.values()]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values) if values else 0
        
        combined_results['average_metrics'] = avg_metrics
        
        # Save combined results
        combined_path = output_dir / "combined_results.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\nCombined results saved to {combined_path}")
        print("Average metrics across experiments:")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    def load_experiment_config(self, config_path: Union[str, Path]) -> List[ExperimentConfig]:
        """Load experiment configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        configs = []
        for exp in config_data['experiments']:
            configs.append(ExperimentConfig(
                model_name=exp['model'],
                model_params=exp.get('params', {}),
                experiment_name=exp.get('name'),
                description=exp.get('description')
            ))
        
        return configs

def main():
    """Main entry point for running determination extraction experiments."""
    parser = argparse.ArgumentParser(description="Run determination extraction experiments")
    parser.add_argument('--config', type=str, help='Path to experiment configuration YAML file')
    parser.add_argument('--model', choices=['ngram', 'basic', 'sparse', 'all'], 
                        help='Single model to run (overrides config file)')
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--test', type=str, help='Path to test data')
    parser.add_argument('--validation', type=str, help='Path to validation data')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--match-threshold', type=float, default=0.75,
                        help='Match threshold for n-gram model')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = DeterminationExperimentRunner()
    
    # Setup data paths if provided
    data_paths = {}
    if args.train:
        data_paths['train'] = Path(args.train)
    if args.test:
        data_paths['test'] = Path(args.test)
    if args.validation:
        data_paths['validation'] = Path(args.validation)
    
    output_dir = Path(args.output) if args.output else None
    
    # Run experiments
    if args.config:
        # Load and run experiments from config file
        configs = runner.load_experiment_config(args.config)
        results = runner.run_experiment(configs, data_paths, output_dir)
    else:
        # Run single model or all models
        if args.model == 'all':
            configs = [
                ExperimentConfig(
                    model_name='ngram',
                    model_params={'match_threshold': args.match_threshold}
                ),
                ExperimentConfig(
                    model_name='basic',
                    model_params={}
                ),
                ExperimentConfig(
                    model_name='sparse',
                    model_params={}
                )
            ]
        else:
            configs = [ExperimentConfig(
                model_name=args.model,
                model_params={'match_threshold': args.match_threshold} if args.model == 'ngram' else {}
            )]
        
        results = runner.run_experiment(configs, data_paths, output_dir)

if __name__ == "__main__":
    main()