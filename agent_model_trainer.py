import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import sys
from io import StringIO
import traceback
from base_agent import BaseAgent
from logger import AgentLogger


class ModelTrainerAgent(BaseAgent):
    
    def __init__(self, logger: AgentLogger):
        super().__init__(
            name="ModelTrainer",
            role="ML Model Coder",
            logger=logger
        )
        self.df = None
        self.target_column = None
        self.iteration = 0
        self.max_iterations = 5
        self.training_history = []
        
    
    def _tool_execute_python_code(self, code: str, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        try:
            import xgboost
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                        f1_score, mean_squared_error, r2_score)
        except ImportError as e:
            return {
                'success': False,
                'metrics': {},
                'stdout': '',
                'stderr': f'Import error: {str(e)}. Please install required packages.',
                'error': str(e)
            }
        
        execution_globals = {
            'pd': pd,
            'np': np,
            'df': df,
            'target_col': target_col,
            'train_test_split': train_test_split,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score,
            'XGBClassifier': xgboost.XGBClassifier,
            'XGBRegressor': xgboost.XGBRegressor,
            '__builtins__': __builtins__
        }
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = StringIO()
        redirected_error = StringIO()
        
        result = {
            'success': False,
            'metrics': {},
            'stdout': '',
            'stderr': '',
            'error': None
        }
        
        try:
            sys.stdout = redirected_output
            sys.stderr = redirected_error
            
            exec(code, execution_globals)
            
            result['success'] = True
            result['metrics'] = execution_globals.get('metrics', {})
            result['stdout'] = redirected_output.getvalue()
            result['stderr'] = redirected_error.getvalue()
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['stderr'] = redirected_error.getvalue() + "\n" + traceback.format_exc()
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return result
    
    
    def process(self, df: pd.DataFrame, previous_report: Dict[str, Any], 
                target_column: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        self.logger.log(self.name, "Process Start", 
                       f"Received engineered data with shape {df.shape}")
        
        self.df = df.copy()
        self.target_column = target_column
        
        self.logger.log(self.name, "Previous Agent Summary", 
                       previous_report.get('summary', 'No summary'))
        
        data_summary = self._analyze_data()
        
        best_metrics = None
        final_code = None
        
        self.logger.log(self.name, "Feedback Loop Start", 
                       f"Maximum iterations: {self.max_iterations}")
        
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            self.logger.log(self.name, f"Iteration {iteration}", 
                          "Generating training code...")
            
            code = self._generate_training_code(data_summary, best_metrics)
            
            self.logger.log(self.name, f"Iteration {iteration}", 
                          "Executing training code...")
            execution_result = self._tool_execute_python_code(
                code, self.df, self.target_column
            )
            
            if not execution_result['success']:
                self.logger.log(self.name, f"Iteration {iteration} - ERROR", 
                              f"Code execution failed: {execution_result['error']}")
                continue
            
            current_metrics = execution_result['metrics']
            self.training_history.append({
                'iteration': iteration,
                'metrics': current_metrics,
                'code': code
            })
            
            self.logger.log(self.name, f"Iteration {iteration} - Metrics", 
                          json.dumps(current_metrics, indent=2))
            
            if best_metrics is None or self._is_better(current_metrics, best_metrics):
                best_metrics = current_metrics
                final_code = code
            
            should_continue = self._should_continue_training(current_metrics, iteration)
            
            if not should_continue:
                self.logger.log(self.name, f"Iteration {iteration} - Decision", 
                              "Performance is satisfactory. Stopping training.")
                break
            else:
                self.logger.log(self.name, f"Iteration {iteration} - Decision", 
                              "Performance can be improved. Continuing...")
        
        if best_metrics is None:
            error_msg = "All training iterations failed. Please check the error logs."
            self.logger.log(self.name, "ERROR", error_msg)
            raise RuntimeError(error_msg)
        
        report = {
            "agent": self.name,
            "total_iterations": len(self.training_history),
            "final_metrics": best_metrics,
            "training_history": self.training_history,
            "summary": self._generate_summary(best_metrics, len(self.training_history)),
            "final_code": final_code
        }
        
        self.save_report(report, "outputs/model_trainer_report.json")
        
        if final_code:
            with open("outputs/final_model_code.py", 'w') as f:
                f.write(final_code)
        
        self.logger.log(self.name, "Process Complete", 
                       f"Best metrics: {best_metrics}")
        
        return best_metrics, report
    
    
    def _analyze_data(self) -> Dict[str, Any]:
        features = [col for col in self.df.columns if col != self.target_column]
        
        analysis = {
            "n_samples": len(self.df),
            "n_features": len(features),
            "features": features,
            "target_column": self.target_column,
            "target_distribution": self.df[self.target_column].value_counts().to_dict(),
            "is_classification": self.df[self.target_column].nunique() < 20,
            "sample_data": self.df.head(3).to_dict()
        }
        
        return analysis
    
    
    def _generate_training_code(self, data_summary: Dict, previous_metrics: Dict = None) -> str:
        
        system_prompt = """You are an expert Machine Learning engineer specialized in XGBoost.

Your task: Generate executable Python code to train and evaluate an XGBoost model.

Requirements:
1. The code must be complete and executable
2. Use train_test_split for validation
3. Train an XGBoost model (XGBClassifier or XGBRegressor)
4. Calculate metrics (accuracy, precision, recall, f1 for classification OR mse, rmse, r2 for regression)
5. Store metrics in a dictionary called 'metrics'
6. The data is already loaded as 'df' and target column as 'target_col'

Available variables and imports (ALREADY IMPORTED - DO NOT IMPORT AGAIN):
- df: pandas DataFrame with the data
- target_col: string, name of the target column
- pd, np: pandas and numpy
- train_test_split: from sklearn.model_selection
- accuracy_score, precision_score, recall_score, f1_score: from sklearn.metrics
- mean_squared_error, r2_score: from sklearn.metrics
- XGBClassifier, XGBRegressor: from xgboost

Output format:
Return ONLY executable Python code, no explanations.
DO NOT include any import statements - they are already available.
The code must end with a 'metrics' dictionary containing the results.

Example structure:
```python
# Prepare data
X = df.drop(columns=[target_col])
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
    'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
    'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
}
```
"""
        
        if previous_metrics is None:
            prompt = f"""Generate code to train a baseline XGBoost model.

DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

This is iteration {self.iteration}. Start with reasonable default hyperparameters.
"""
        else:
            prompt = f"""The previous model achieved these metrics:
{json.dumps(previous_metrics, indent=2)}

Generate IMPROVED code with different hyperparameters to get better performance.

DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

This is iteration {self.iteration}. Try adjusting:
- learning_rate (try values between 0.01 and 0.3)
- max_depth (try values between 3 and 10)
- n_estimators (try values between 50 and 300)
- subsample (try values between 0.6 and 1.0)
- colsample_bytree (try values between 0.6 and 1.0)

Focus on improving the metrics, especially accuracy/f1 for classification or r2 for regression.
"""
        
        llm_response = self.call_llm(prompt, system_prompt)
        
        code = self._extract_code_from_response(llm_response)
        
        self.logger.log(self.name, f"Iteration {self.iteration} - Generated Code", 
                       code[:300] + "..." if len(code) > 300 else code)
        
        return code
    
    def _extract_code_from_response(self, response: str) -> str:
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()
        
        return code
    
    def _is_better(self, current_metrics: Dict, best_metrics: Dict) -> bool:
        """determine if current metrics are better than best metrics."""
        # for classification, use F1 score
        if 'f1' in current_metrics:
            return current_metrics.get('f1', 0) > best_metrics.get('f1', 0)
        # for regression, use R2 score
        elif 'r2' in current_metrics:
            return current_metrics.get('r2', -999) > best_metrics.get('r2', -999)
        # fallback to accuracy
        else:
            return current_metrics.get('accuracy', 0) > best_metrics.get('accuracy', 0)
    
    def _should_continue_training(self, current_metrics: Dict, iteration: int) -> bool:
        if iteration >= self.max_iterations:
            return False
        
        decision_prompt = f"""You have trained an XGBoost model with these results:

CURRENT METRICS (Iteration {iteration}):
{json.dumps(current_metrics, indent=2)}

TRAINING HISTORY:
{json.dumps([h['metrics'] for h in self.training_history], indent=2)}

Decision: Should we continue training with different hyperparameters, or is this performance good enough?

Consider:
- Is the accuracy/f1/r2 score satisfactory? (>0.75 is usually good)
- Are we seeing improvement across iterations?
- Have we plateaued (no improvement in last 2 iterations)?

Respond with ONLY a JSON object:
{{
  "continue": true or false,
  "reasoning": "Your explanation"
}}
"""
        
        system_prompt = """You are an ML expert making decisions about model training.
Respond with ONLY a JSON object as specified. No other text."""
        
        llm_response = self.call_llm(decision_prompt, system_prompt)
        
        # parse decision
        try:
            response_text = llm_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            decision = json.loads(response_text)
            
            self.logger.log(self.name, f"Iteration {iteration} - LLM Decision Reasoning", 
                          decision.get("reasoning", "No reasoning provided"))
            
            return decision.get("continue", False)
            
        except json.JSONDecodeError:
            # fallback: continue if performance is poor
            if 'f1' in current_metrics:
                return current_metrics.get('f1', 0) < 0.75
            elif 'r2' in current_metrics:
                return current_metrics.get('r2', 0) < 0.75
            else:
                return current_metrics.get('accuracy', 0) < 0.75
    
    def _generate_summary(self, metrics: Dict, iterations: int) -> str:
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        return f"Trained XGBoost model over {iterations} iterations. Final metrics: {metric_str}"