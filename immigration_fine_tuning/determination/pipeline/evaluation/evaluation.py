import pandas as pd
import json
import os
from pathlib import Path

def load_data(base_path=None):
    """Load train and test data from the specified path."""
    if base_path is None:
        # Use direct project path instead of environment variable
        base_path = Path(__file__).parent.parent.parent.parent / "data" / "merged"
    else:
        base_path = Path(base_path)
    
    train_path = base_path / "train_enriched.csv"
    test_path = base_path / "test_enriched.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Could not find train/test files in {base_path}")
    
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    
    return train_df, test_df

def create_validation_set(train_df, test_df):
    """Create validation set based on determination counts."""
    # Extract validation samples from train where determination_count >= 4
    train_validation = train_df[train_df['determination_count'] >= 4].copy()
    
    # Extract validation samples from test where determination_count >= 3
    test_validation = test_df[test_df['determination_count'] >= 3].copy()
    
    # Combine into a single validation set
    validation_set = pd.concat([train_validation, test_validation], ignore_index=True)
    
    # Keep only relevant columns for determination extraction
    relevant_columns = [
        'decisionID', 
        'full_text', 
        'extracted_sentences_determination', 
        'determination_count'
    ]
    
    # Filter to only keep relevant columns that exist in the dataframe
    existing_columns = [col for col in relevant_columns if col in validation_set.columns]
    validation_set = validation_set[existing_columns]
    
    print(f"Created validation set with {len(validation_set)} cases")
    print(f"  - {len(train_validation)} from train")
    print(f"  - {len(test_validation)} from test")
    print(f"  - Kept {len(existing_columns)} relevant columns: {existing_columns}")
    
    return validation_set

def evaluate_extraction_model(model, validation_set):
    """Evaluate the extraction model on the validation set."""
    results = []
    
    for idx, case in validation_set.iterrows():
        # Get original case text
        original_text = case['full_text']
        
        # Get ground truth determinations (if available)
        ground_truth = []
        if 'extracted_sentences_determination' in case and isinstance(case['extracted_sentences_determination'], str):
            # Handle different formats of determination storage
            if case['extracted_sentences_determination'].startswith('[') or case['extracted_sentences_determination'].startswith('{'):
                try:
                    ground_truth = json.loads(case['extracted_sentences_determination'])
                except:
                    ground_truth = [case['extracted_sentences_determination']]
            else:
                ground_truth = [case['extracted_sentences_determination']]
        
        # Process with model
        model_output = model.process_case(original_text)
        
        # Calculate metrics
        metrics = calculate_metrics(model_output, ground_truth, case['determination_count'])
        
        # Store results
        results.append({
            'case_id': case.get('decisionID', idx),
            'metrics': metrics,
            'model_output': model_output
        })
    
    # Aggregate results
    aggregate_metrics = aggregate_results(results)
    
    return results, aggregate_metrics

def calculate_metrics(model_output, ground_truth, expected_count):
    """Calculate evaluation metrics for a single case."""
    extracted_count = len(model_output['extracted_determinations'])
    
    # Basic count-based metrics
    metrics = {
        'extracted_count': extracted_count,
        'expected_count': expected_count,
        'count_ratio': extracted_count / max(1, expected_count)
    }
    
    # More sophisticated precision/recall metrics are commented out for now
    # as they require more complex matching logic
    """
    # If we have ground truth, calculate precision/recall
    if ground_truth:
        # This is simplified - in practice you'd need text matching logic
        matched = 0
        for gt in ground_truth:
            for ext in model_output['extracted_determinations']:
                if gt.lower() in ext['text'].lower() or ext['text'].lower() in gt.lower():
                    matched += 1
                    break
        
        metrics['precision'] = matched / max(1, extracted_count)
        metrics['recall'] = matched / len(ground_truth)
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / max(1, metrics['precision'] + metrics['recall'])
    """
    
    return metrics

def aggregate_results(results):
    """Aggregate metrics across all evaluated cases."""
    # Extract all metrics
    all_metrics = [r['metrics'] for r in results]
    
    # Calculate averages
    agg = {}
    for key in all_metrics[0].keys():
        agg[f'avg_{key}'] = sum(m[key] for m in all_metrics if key in m) / len(all_metrics)
    
    # Count cases where we extracted at least as many as expected
    agg['cases_meeting_expected'] = sum(1 for m in all_metrics if m['count_ratio'] >= 1)
    agg['total_cases'] = len(all_metrics)
    agg['pct_meeting_expected'] = agg['cases_meeting_expected'] / agg['total_cases']
    
    return agg

def analyze_challenging_cases(results, validation_df, threshold_ratio=0.5, max_cases=20):
    """
    Identify cases where the model found significantly fewer determinations than expected.
    
    Args:
        results: Evaluation results from run_evaluation
        validation_df: Validation dataframe with original texts
        threshold_ratio: Threshold ratio below which a case is considered challenging
        max_cases: Maximum number of cases to include in the analysis
        
    Returns:
        List of challenging cases with analysis data
    """
    # Convert validation_df to dictionary for faster lookup
    case_texts = {}
    ground_truth_determinations = {}
    
    for _, row in validation_df.iterrows():
        case_id = str(row.get('decisionID', ''))
        if not case_id:
            continue
            
        # Get case text
        if 'cleaned_text' in row and isinstance(row['cleaned_text'], str):
            case_texts[case_id] = row['cleaned_text']
        elif 'full_text' in row and isinstance(row['full_text'], str):
            case_texts[case_id] = row['full_text']
            
        # Get ground truth determinations
        if 'extracted_sentences_determination' in row and isinstance(row['extracted_sentences_determination'], str):
            try:
                # Try parsing as JSON first
                if row['extracted_sentences_determination'].startswith('[') or row['extracted_sentences_determination'].startswith('{'):
                    parsed = json.loads(row['extracted_sentences_determination'].replace("'", '"'))
                    if isinstance(parsed, list):
                        ground_truth_determinations[case_id] = [
                            item if isinstance(item, str) else
                            item.get('text', str(item)) if isinstance(item, dict) else
                            str(item) for item in parsed
                        ]
                    elif isinstance(parsed, dict) and 'text' in parsed:
                        ground_truth_determinations[case_id] = [parsed['text']]
                    else:
                        ground_truth_determinations[case_id] = [row['extracted_sentences_determination']]
                else:
                    # Treat as plain text
                    ground_truth_determinations[case_id] = [row['extracted_sentences_determination']]
            except:
                # If parsing fails, treat as plain text
                ground_truth_determinations[case_id] = [row['extracted_sentences_determination']]
    
    # Find challenging cases
    challenging_cases = []
    
    for r in results:
        case_id = str(r['case_id'])
        metrics = r['metrics']
        expected = metrics['expected_count']
        extracted = metrics['extracted_count']
        ratio = metrics['count_ratio']
        
        # Cases with low ratio of extracted to expected determinations
        if expected > 1 and ratio < threshold_ratio:
            case_text = case_texts.get(case_id, "Text not available")
            
            # Get the extracted determinations
            extracted_determinations = []
            if 'model_output' in r and 'extracted_determinations' in r['model_output']:
                for det in r['model_output']['extracted_determinations']:
                    extracted_determinations.append({
                        'text': det['text'],
                        'score': det.get('score', 0),
                        'match_type': det.get('match_type', 'unknown')
                    })
            
            # Get expected (ground truth) determinations
            expected_determinations = ground_truth_determinations.get(case_id, [])
            
            # Identify which expected determinations were NOT found
            extracted_texts = set(det['text'].strip().lower() for det in extracted_determinations)
            missing_determinations = []
            
            for expected_det in expected_determinations:
                if not expected_det or not isinstance(expected_det, str):
                    continue
                    
                found = False
                expected_text = expected_det.strip().lower()
                
                # Check if this expected determination was found (exact or fuzzy match)
                for extracted_text in extracted_texts:
                    # Exact match
                    if expected_text == extracted_text:
                        found = True
                        break
                    
                    # Substring match
                    if expected_text in extracted_text or extracted_text in expected_text:
                        found = True
                        break
                
                if not found:
                    missing_determinations.append(expected_det)
            
            challenging_cases.append({
                'case_id': case_id,
                'expected_count': expected,
                'extracted_count': extracted,
                'ratio': ratio,
                'extracted_determinations': extracted_determinations,
                'expected_determinations': expected_determinations,
                'missing_determinations': missing_determinations,
                'case_text': case_text[:2000] + "..." if len(case_text) > 2000 else case_text
            })
    
    # Sort by the gap between expected and extracted (largest first)
    challenging_cases.sort(key=lambda x: (x['expected_count'] - x['extracted_count']), reverse=True)
    
    # Take the top N most challenging cases
    challenging_cases = challenging_cases[:max_cases]
    
    return challenging_cases

def save_analysis_reports(challenging_cases, output_dir, prefix="challenging_cases"):
    """
    Save analysis of challenging cases to JSON and HTML files.
    
    Args:
        challenging_cases: List of challenging cases from analyze_challenging_cases
        output_dir: Directory to save reports
        prefix: Prefix for the output filenames
        
    Returns:
        Tuple of (json_path, html_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / f"{prefix}.json"
    html_path = output_dir / f"{prefix}.html"
    
    # Save JSON report
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_summary': {
                'total_challenging_cases': len(challenging_cases),
                'max_cases_shown': len(challenging_cases)
            },
            'challenging_cases': challenging_cases
        }, f, indent=2)
    
    # Create HTML report
    html = _create_html_report(challenging_cases)
    
    # Save HTML report
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return json_path, html_path

def _create_html_report(challenging_cases):
    """Create an HTML report for easier viewing of challenging cases."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Determination Extraction Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .case { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .case-header { background-color: #f8f9fa; padding: 10px; margin-bottom: 15px; }
            .extracted { background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }
            .expected { background-color: #f8f4e8; padding: 10px; margin: 10px 0; border-left: 4px solid #f39c12; }
            .missing { background-color: #f8e8e8; padding: 10px; margin: 10px 0; border-left: 4px solid #e74c3c; }
            .case-text { background-color: #f9f9f9; padding: 15px; border: 1px solid #eee; 
                         height: 200px; overflow: auto; font-family: monospace; white-space: pre-wrap; }
            .metrics { font-weight: bold; color: #e74c3c; }
            .comparison { display: flex; flex-wrap: wrap; gap: 20px; }
            .comparison-column { flex: 1; min-width: 300px; }
            .nav { position: sticky; top: 0; background: white; padding: 10px; border-bottom: 1px solid #ddd; }
            .legend { margin: 10px 0 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            .legend span { margin-right: 20px; }
        </style>
    </head>
    <body>
        <div class="nav">
            <h1>Challenging Cases in Determination Extraction</h1>
            <div class="legend">
                <span style="color: #3498db;">■</span> Extracted
                <span style="color: #f39c12;">■</span> Expected
                <span style="color: #e74c3c;">■</span> Missing (Expected but not found)
            </div>
        </div>
        <p>These cases had significantly fewer extracted determinations than expected.</p>
    """
    
    for i, case in enumerate(challenging_cases):
        html += f"""
        <div class="case" id="case-{i+1}">
            <div class="case-header">
                <h2>Case {i+1}: ID {case['case_id']}</h2>
                <p class="metrics">Expected: {case['expected_count']} | Extracted: {case['extracted_count']} | 
                   Ratio: {case['ratio']:.2f}</p>
            </div>
            
            <div class="comparison">
                <div class="comparison-column">
                    <h3>Extracted Determinations ({len(case['extracted_determinations'])})</h3>
        """
        
        if case['extracted_determinations']:
            for det in case['extracted_determinations']:
                html += f"""
                <div class="extracted">
                    <p><strong>Score:</strong> {det.get('score', 'N/A')} | <strong>Match type:</strong> {det.get('match_type', 'unknown')}</p>
                    <p>{det['text']}</p>
                </div>
                """
        else:
            html += "<p>No determinations extracted.</p>"
        
        html += f"""
                </div>
                <div class="comparison-column">
                    <h3>Missing Determinations ({len(case.get('missing_determinations', []))})</h3>
        """
        
        if case.get('missing_determinations'):
            for det in case['missing_determinations']:
                html += f"""
                <div class="missing">
                    <p>{det}</p>
                </div>
                """
        else:
            html += "<p>No missing determinations (all expected determinations were found).</p>"
        
        html += """
                </div>
            </div>
        """
        
        # Add a section for all expected determinations
        html += f"""
            <div>
                <h3>All Expected Determinations ({len(case.get('expected_determinations', []))})</h3>
        """
        
        if case.get('expected_determinations'):
            for det in case['expected_determinations']:
                html += f"""
                <div class="expected">
                    <p>{det}</p>
                </div>
                """
        else:
            html += "<p>No expected determinations available.</p>"
        
        html += f"""
            </div>
            
            <h3>Case Text (first portion):</h3>
            <div class="case-text">{case['case_text'].replace('<', '&lt;').replace('>', '&gt;')}</div>
        </div>
        """
    
    html += """
    <script>
        // Add a simple navigation for jumping between cases
        document.addEventListener('DOMContentLoaded', function() {
            const nav = document.querySelector('.nav');
            const navContent = document.createElement('div');
            navContent.style.marginTop = '10px';
            
            for (let i = 1; i <= """ + str(len(challenging_cases)) + """; i++) {
                const link = document.createElement('a');
                link.href = '#case-' + i;
                link.textContent = 'Case ' + i;
                link.style.marginRight = '15px';
                link.style.textDecoration = 'none';
                link.style.padding = '5px 10px';
                link.style.backgroundColor = '#f8f9fa';
                link.style.borderRadius = '5px';
                navContent.appendChild(link);
            }
            
            nav.appendChild(navContent);
        });
    </script>
    </body>
    </html>
    """
    
    return html

def analyze_overextraction_cases(results, validation_df, min_extra_count=2, max_cases=20):
    """
    Identify cases where the model extracted significantly more determinations than expected.
    
    Args:
        results: Evaluation results from run_evaluation
        validation_df: Validation dataframe with original texts
        min_extra_count: Minimum number of extra determinations to consider as significant
        max_cases: Maximum number of cases to include in the analysis
        
    Returns:
        List of over-extraction cases with analysis data
    """
    # Convert validation_df to dictionary for faster lookup
    case_texts = {}
    ground_truth_determinations = {}
    
    for _, row in validation_df.iterrows():
        case_id = str(row.get('decisionID', ''))
        if not case_id:
            continue
            
        # Get case text
        if 'cleaned_text' in row and isinstance(row['cleaned_text'], str):
            case_texts[case_id] = row['cleaned_text']
        elif 'full_text' in row and isinstance(row['full_text'], str):
            case_texts[case_id] = row['full_text']
            
        # Get ground truth determinations
        if 'extracted_sentences_determination' in row and isinstance(row['extracted_sentences_determination'], str):
            try:
                # Try parsing as JSON first
                if row['extracted_sentences_determination'].startswith('[') or row['extracted_sentences_determination'].startswith('{'):
                    parsed = json.loads(row['extracted_sentences_determination'].replace("'", '"'))
                    if isinstance(parsed, list):
                        ground_truth_determinations[case_id] = [
                            item if isinstance(item, str) else
                            item.get('text', str(item)) if isinstance(item, dict) else
                            str(item) for item in parsed
                        ]
                    elif isinstance(parsed, dict) and 'text' in parsed:
                        ground_truth_determinations[case_id] = [parsed['text']]
                    else:
                        ground_truth_determinations[case_id] = [row['extracted_sentences_determination']]
                else:
                    # Treat as plain text
                    ground_truth_determinations[case_id] = [row['extracted_sentences_determination']]
            except:
                # If parsing fails, treat as plain text
                ground_truth_determinations[case_id] = [row['extracted_sentences_determination']]
    
    # Find over-extraction cases
    overextraction_cases = []
    
    for r in results:
        case_id = str(r['case_id'])
        metrics = r['metrics']
        expected = metrics['expected_count']
        extracted = metrics['extracted_count']
        
        # Skip cases with zero or few expected determinations
        if expected < 1:
            continue
            
        # Cases with significantly more extractions than expected
        if extracted > expected + min_extra_count:
            case_text = case_texts.get(case_id, "Text not available")
            
            # Get the extracted determinations
            extracted_determinations = []
            if 'model_output' in r and 'extracted_determinations' in r['model_output']:
                for det in r['model_output']['extracted_determinations']:
                    extracted_determinations.append({
                        'text': det['text'],
                        'score': det.get('score', 0),
                        'match_type': det.get('match_type', 'unknown')
                    })
            
            # Get expected (ground truth) determinations
            expected_determinations = ground_truth_determinations.get(case_id, [])
            
            # Identify which extracted determinations were NOT in the ground truth
            expected_texts = [det.lower().strip() for det in expected_determinations if det]
            extra_determinations = []
            
            for extracted_det in extracted_determinations:
                extracted_text = extracted_det['text'].lower().strip()
                
                # Check if this extracted determination is not in the expected set
                found = False
                for expected_text in expected_texts:
                    # Exact match
                    if extracted_text == expected_text:
                        found = True
                        break
                    
                    # Substring match
                    if extracted_text in expected_text or expected_text in extracted_text:
                        found = True
                        break
                
                if not found:
                    extra_determinations.append(extracted_det)
            
            overextraction_cases.append({
                'case_id': case_id,
                'expected_count': expected,
                'extracted_count': extracted,
                'extra_count': extracted - expected,
                'ratio': extracted / expected if expected > 0 else float('inf'),
                'extracted_determinations': extracted_determinations,
                'expected_determinations': expected_determinations,
                'extra_determinations': extra_determinations,
                'case_text': case_text[:2000] + "..." if len(case_text) > 2000 else case_text
            })
    
    # Sort by the number of extra determinations (largest first)
    overextraction_cases.sort(key=lambda x: x['extra_count'], reverse=True)
    
    # Take the top N most over-extracting cases
    overextraction_cases = overextraction_cases[:max_cases]
    
    return overextraction_cases

def create_overextraction_html_report(overextraction_cases, output_path):
    """Create an HTML report for easier viewing of over-extraction cases."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Over-Extraction Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .case { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .case-header { background-color: #f8f9fa; padding: 10px; margin-bottom: 15px; }
            .extracted { background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }
            .expected { background-color: #f8f4e8; padding: 10px; margin: 10px 0; border-left: 4px solid #f39c12; }
            .extra { background-color: #e8f8ec; padding: 10px; margin: 10px 0; border-left: 4px solid #2ecc71; }
            .case-text { background-color: #f9f9f9; padding: 15px; border: 1px solid #eee; 
                         height: 200px; overflow: auto; font-family: monospace; white-space: pre-wrap; }
            .metrics { font-weight: bold; color: #e74c3c; }
            .comparison { display: flex; flex-wrap: wrap; gap: 20px; }
            .comparison-column { flex: 1; min-width: 300px; }
            .nav { position: sticky; top: 0; background: white; padding: 10px; border-bottom: 1px solid #ddd; }
            .legend { margin: 10px 0 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            .legend span { margin-right: 20px; }
        </style>
    </head>
    <body>
        <div class="nav">
            <h1>Over-Extraction Cases in Determination Analysis</h1>
            <div class="legend">
                <span style="color: #3498db;">■</span> All Extracted
                <span style="color: #f39c12;">■</span> Expected (Benchmark)
                <span style="color: #2ecc71;">■</span> Extra (Not in Benchmark)
            </div>
        </div>
        <p>These cases had significantly more extracted determinations than expected in the benchmark.</p>
    """
    
    for i, case in enumerate(overextraction_cases):
        html += f"""
        <div class="case" id="case-{i+1}">
            <div class="case-header">
                <h2>Case {i+1}: ID {case['case_id']}</h2>
                <p class="metrics">Expected: {case['expected_count']} | Extracted: {case['extracted_count']} | 
                   Extra: {case['extra_count']} | Ratio: {case['ratio']:.2f}</p>
            </div>
            
            <div class="comparison">
                <div class="comparison-column">
                    <h3>Expected Determinations ({len(case['expected_determinations'])})</h3>
        """
        
        if case['expected_determinations']:
            for det in case['expected_determinations']:
                html += f"""
                <div class="expected">
                    <p>{det}</p>
                </div>
                """
        else:
            html += "<p>No expected determinations available.</p>"
        
        html += f"""
                </div>
                <div class="comparison-column">
                    <h3>Extra Determinations ({len(case['extra_determinations'])})</h3>
        """
        
        if case['extra_determinations']:
            for det in case['extra_determinations']:
                html += f"""
                <div class="extra">
                    <p><strong>Score:</strong> {det.get('score', 'N/A')} | <strong>Match type:</strong> {det.get('match_type', 'unknown')}</p>
                    <p>{det['text']}</p>
                </div>
                """
        else:
            html += "<p>No extra determinations (all extractions were in the benchmark).</p>"
        
        html += """
                </div>
            </div>
        """
        
        # Add a section for all extracted determinations
        html += f"""
            <div>
                <h3>All Extracted Determinations ({len(case.get('extracted_determinations', []))})</h3>
        """
        
        if case.get('extracted_determinations'):
            for det in case['extracted_determinations']:
                html += f"""
                <div class="extracted">
                    <p><strong>Score:</strong> {det.get('score', 'N/A')} | <strong>Match type:</strong> {det.get('match_type', 'unknown')}</p>
                    <p>{det['text']}</p>
                </div>
                """
        else:
            html += "<p>No extracted determinations available.</p>"
        
        html += f"""
            </div>
            
            <h3>Case Text (first portion):</h3>
            <div class="case-text">{case['case_text'].replace('<', '&lt;').replace('>', '&gt;')}</div>
        </div>
        """
    
    html += """
    <script>
        // Add a simple navigation for jumping between cases
        document.addEventListener('DOMContentLoaded', function() {
            const nav = document.querySelector('.nav');
            const navContent = document.createElement('div');
            navContent.style.marginTop = '10px';
            
            for (let i = 1; i <= """ + str(len(overextraction_cases)) + """; i++) {
                const link = document.createElement('a');
                link.href = '#case-' + i;
                link.textContent = 'Case ' + i;
                link.style.marginRight = '15px';
                link.style.textDecoration = 'none';
                link.style.padding = '5px 10px';
                link.style.backgroundColor = '#f8f9fa';
                link.style.borderRadius = '5px';
                navContent.appendChild(link);
            }
            
            nav.appendChild(navContent);
        });
    </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

def run_evaluation(model, validation_file=None, output_dir=None, analyze_failures=True, analyze_overextraction=True):
    """
    Run evaluation on a model using the validation set.
    
    Args:
        model: The model to evaluate
        validation_file: Path to validation CSV file (if None, uses default)
        output_dir: Directory to save analysis reports (if None, uses model output dir)
        analyze_failures: Whether to generate analysis of challenging cases
        analyze_overextraction: Whether to generate analysis of over-extraction cases
    
    Returns:
        Tuple of (detailed_results, aggregate_metrics, analysis_paths)
    """
    if validation_file is None:
        # Update default validation file path to be in pipeline/data
        validation_file = Path(__file__).parent.parent / "data" / "validation_set.csv"
    
    if output_dir is None:
        # Use pipeline/results/evaluation as default output directory
        output_dir = Path(__file__).parent.parent / "results" / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_set = pd.read_csv(validation_file)
    print(f"Loaded validation set with {len(validation_set)} cases")
    
    results, metrics = evaluate_extraction_model(model, validation_set)
    
    print("\nAggregate Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Generate analysis reports if requested
    analysis_paths = {}
    
    if analyze_failures:
        challenging_cases = analyze_challenging_cases(results, validation_set)
        json_path, html_path = save_analysis_reports(challenging_cases, output_dir)
        
        analysis_paths['challenging'] = {'json': str(json_path), 'html': str(html_path)}
        
        print(f"\nSaved analysis of {len(challenging_cases)} challenging cases to:")
        print(f"  - JSON: {json_path}")
        print(f"  - HTML: {html_path}")
    
    if analyze_overextraction:
        overextraction_cases = analyze_overextraction_cases(results, validation_set)
        json_path = output_dir / "overextraction_cases.json"
        html_path = output_dir / "overextraction_cases.html"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_summary': {
                    'total_overextraction_cases': len(overextraction_cases),
                    'max_cases_shown': len(overextraction_cases)
                },
                'overextraction_cases': overextraction_cases
            }, f, indent=2)
        
        # Create HTML report
        create_overextraction_html_report(overextraction_cases, html_path)
        
        analysis_paths['overextraction'] = {'json': str(json_path), 'html': str(html_path)}
        
        print(f"\nSaved analysis of {len(overextraction_cases)} over-extraction cases to:")
        print(f"  - JSON: {json_path}")
        print(f"  - HTML: {html_path}")
    
    return results, metrics, analysis_paths

def main():
    # Get project directories
    script_dir = Path(__file__).resolve().parent  # evaluation/
    pipeline_dir = script_dir.parent  # pipeline/
    
    # # Save validation set in our pipeline/data directory
    # validation_file = pipeline_dir / "data" / "validation_set.csv"
    
    # # Create the validation set
    # train_df, test_df = load_data()  # Will use project path
    # validation_set = create_validation_set(train_df, test_df)
    
    # # Save validation set in pipeline/data directory
    # validation_set.to_csv(validation_file, index=False)
    # print(f"Saved validation set to {validation_file}")

    print("This module provides evaluation functions for determination extraction models.")
    print("To run an evaluation, import run_evaluation and pass your model.")
    print("To recreate the validation set, uncomment the relevant lines in main().")

if __name__ == "__main__":
    main()
