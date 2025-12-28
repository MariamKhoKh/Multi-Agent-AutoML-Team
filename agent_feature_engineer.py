import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from base_agent import BaseAgent, ToolRegistry
from logger import AgentLogger


class FeatureEngineerAgent(BaseAgent):
    
    def __init__(self, logger: AgentLogger):
        super().__init__(
            name="FeatureEngineer",
            role="Feature Architect",
            logger=logger
        )
        self.df = None
        self.target_column = None
        self.tool_registry = ToolRegistry()
        self._register_tools()
        
    def _register_tools(self):
        self.tool_registry.register(
            "create_interaction",
            "Creates a new column using mathematical expressions between columns",
            {
                "df": "The dataframe",
                "new_col": "Name for the new column",
                "expression": "Python expression like 'df[col1] / df[col2]' or 'df[col1] * df[col2]'"
            }
        )
        self.tool_registry.register(
            "encode_categorical",
            "Encodes categorical columns into numeric format",
            {
                "df": "The dataframe",
                "col": "Column name to encode",
                "method": "Either 'label' (ordinal) or 'onehot' (binary columns)"
            }
        )
        self.tool_registry.register(
            "correlation_analysis",
            "Analyzes correlation between features and target",
            {
                "df": "The dataframe",
                "target": "Target column name"
            }
        )
        self.tool_registry.register(
            "select_top_features",
            "Keeps only the k most predictive features",
            {
                "df": "The dataframe",
                "target": "Target column name",
                "k": "Number of top features to keep"
            }
        )
    
    def _tool_create_interaction(self, df: pd.DataFrame, new_col: str, expression: str) -> pd.DataFrame:
        try:
            df[new_col] = eval(expression)
            df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
            df[new_col] = df[new_col].fillna(df[new_col].median())
            return df
        except Exception as e:
            raise ValueError(f"Error creating interaction: {e}")
    
    def _tool_encode_categorical(self, df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        
        if method == "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return df
    
    def _tool_correlation_analysis(self, df: pd.DataFrame, target: str) -> str:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target in numeric_cols:
            numeric_cols.remove(target)
        
        correlations = {}
        for col in numeric_cols:
            corr = df[col].corr(df[target])
            correlations[col] = round(corr, 4) if not pd.isna(corr) else 0.0
        
        sorted_corrs = dict(sorted(correlations.items(), 
                                  key=lambda x: abs(x[1]), 
                                  reverse=True))
        
        result = {
            "correlations": sorted_corrs,
            "high_correlation": [k for k, v in sorted_corrs.items() if abs(v) > 0.5],
            "low_correlation": [k for k, v in sorted_corrs.items() if abs(v) < 0.1]
        }
        
        return json.dumps(result, indent=2)
    
    def _tool_select_top_features(self, df: pd.DataFrame, target: str, k: int) -> pd.DataFrame:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        
        X = df.drop(columns=[target])
        y = df[target]
        
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]
        
        if len(numeric_features) == 0:
            raise ValueError("No numeric features available for selection")
        
        k = min(k, len(numeric_features)) 
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_numeric, y)
        
        selected_features = numeric_features[selector.get_support()].tolist()
        selected_features.append(target) 
        
        return df[selected_features]
    
    
    def process(self, df: pd.DataFrame, previous_report: Dict[str, Any], 
                target_column: str) -> Tuple[str, Dict[str, Any]]:
        """
        returns: (path_to_engineered_data, report_dict)
        """
        self.logger.log(self.name, "Process Start", 
                       f"Received clean data with shape {df.shape}")
        
        self.df = df.copy()
        self.target_column = target_column
        original_shape = self.df.shape
        
        self.logger.log(self.name, "Previous Agent Summary", 
                       previous_report.get('summary', 'No summary'))
        
        analysis = self._analyze_features()
        
        engineering_prompt = self._build_engineering_prompt(analysis, previous_report)
        llm_response = self.call_llm(engineering_prompt, self._get_system_prompt())
        
        actions_taken = self._execute_llm_decisions(llm_response)
        
        output_path = "outputs/engineered_data.csv"
        self.df.to_csv(output_path, index=False)
        self.logger.log(self.name, "Data Saved", f"Engineered data saved to {output_path}")
        
        report = {
            "agent": self.name,
            "original_shape": original_shape,
            "final_shape": self.df.shape,
            "actions_taken": actions_taken,
            "summary": self._generate_summary(actions_taken),
            "final_features": list(self.df.columns)
        }
        
        self.save_report(report, "outputs/feature_engineer_report.json")
        
        self.logger.log(self.name, "Process Complete", 
                       f"Shape: {original_shape} -> {self.df.shape}")
        
        return output_path, report
    
    
    def _analyze_features(self) -> Dict[str, Any]:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        analysis = {
            "total_features": len(self.df.columns) - 1,  # exclude target
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "shape": self.df.shape,
            "sample_stats": {}
        }
        
        # get basic stats for numeric features
        for col in numeric_cols[:3]: 
            analysis["sample_stats"][col] = {
                "mean": float(self.df[col].mean()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max())
            }
        
        return analysis
    
    def _get_system_prompt(self) -> str:
        return f"""You are the Feature Engineer Agent, an expert in creating predictive features.

Your role: Receive cleaned data and create new features to maximize model performance.

Available Tools:
{self.tool_registry.get_tool_descriptions()}

Your task:
1. Analyze the current features
2. Create new interaction features (ratios, products, etc.)
3. Encode categorical variables appropriately
4. Select the most predictive features
5. Output your decisions in a structured format

Guidelines:
- Create meaningful interactions (e.g., income/age ratio, area*price)
- Use label encoding for ordinal categories, one-hot for nominal
- Remove features with very low correlation to target (<0.05)
- Keep feature count reasonable (prefer 10-15 final features)

Output Format (JSON):
{{
  "reasoning": "Your strategy and why these features will help",
  "actions": [
    {{"action": "create_interaction", "new_col": "income_per_age", "expression": "df['income'] / (df['age'] + 1)", "reason": "Normalize income by age"}},
    {{"action": "encode_categorical", "column": "category", "method": "onehot", "reason": "Nominal variable needs one-hot encoding"}},
    {{"action": "correlation_analysis", "reason": "Check which features correlate with target"}},
    {{"action": "select_top_features", "k": 10, "reason": "Keep only most predictive features"}}
  ]
}}

Be creative but practical. Focus on features that make logical sense."""
    
    def _build_engineering_prompt(self, analysis: Dict, previous_report: Dict) -> str:
        """Build the feature engineering prompt."""
        return f"""You have received cleaned data from the Data Cleaner agent.

PREVIOUS AGENT'S WORK:
{previous_report.get('summary', 'Data cleaning completed')}

CURRENT FEATURE ANALYSIS:
{json.dumps(analysis, indent=2)}

TARGET COLUMN: {self.target_column}

Based on this information, what feature engineering actions should you perform?
Think about:
- What new features could be informative?
- Which categorical variables need encoding?
- Are there redundant features to remove?

Provide your response in the JSON format specified."""
    
    def _execute_llm_decisions(self, llm_response: str) -> List[str]:
        self.logger.log(self.name, "LLM Decision", "Parsing feature engineering decisions")
        
        actions_taken = []
        
        try:
            response_text = llm_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            decisions = json.loads(response_text)
            
            self.logger.log(self.name, "LLM Reasoning", 
                          decisions.get("reasoning", "No reasoning provided"))
            
            for action_spec in decisions.get("actions", []):
                action_type = action_spec.get("action")
                reason = action_spec.get("reason", "No reason provided")
                
                self.logger.log(self.name, f"Action: {action_type}", reason)
                
                if action_type == "create_interaction":
                    new_col = action_spec.get("new_col")
                    expression = action_spec.get("expression")
                    self.df = self.execute_tool("create_interaction", 
                                               df=self.df, 
                                               new_col=new_col, 
                                               expression=expression)
                    actions_taken.append(f"Created feature '{new_col}': {reason}")
                
                elif action_type == "encode_categorical":
                    column = action_spec.get("column")
                    method = action_spec.get("method", "label")
                    self.df = self.execute_tool("encode_categorical", 
                                               df=self.df, 
                                               col=column, 
                                               method=method)
                    actions_taken.append(f"Encoded '{column}' with {method}: {reason}")
                
                elif action_type == "correlation_analysis":
                    result = self.execute_tool("correlation_analysis", 
                                              df=self.df, 
                                              target=self.target_column)
                    actions_taken.append(f"Analyzed correlations: {reason}")
                    self.logger.log(self.name, "Correlation Results", result[:500])
                
                elif action_type == "select_top_features":
                    k = action_spec.get("k", 10)
                    self.df = self.execute_tool("select_top_features", 
                                               df=self.df, 
                                               target=self.target_column, 
                                               k=k)
                    actions_taken.append(f"Selected top {k} features: {reason}")
        
        except json.JSONDecodeError as e:
            self.logger.log(self.name, "ERROR", f"Failed to parse LLM response: {e}")
            self.logger.log(self.name, "Raw Response", llm_response[:500])
            actions_taken.append("ERROR: Could not parse LLM decisions, performed basic encoding")
            self._fallback_engineering()
        
        return actions_taken
    
    def _fallback_engineering(self):
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        for col in categorical_cols:
            self.df = self._tool_encode_categorical(self.df, col, "label")
            self.logger.log(self.name, "Fallback", f"Label encoded {col}")
    
    def _generate_summary(self, actions: List[str]) -> str:
        if not actions:
            return "No feature engineering was necessary."
        
        summary = f"Performed {len(actions)} feature engineering actions: "
        summary += "; ".join(actions[:3])
        if len(actions) > 3:
            summary += f"; and {len(actions) - 3} more actions."
        
        return summary