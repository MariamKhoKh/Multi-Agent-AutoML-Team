from logger import AgentLogger
from handoff import Handoff
from datetime import datetime
from pathlib import Path
from agent_data_cleaner import DataCleanerAgent
from agent_feature_engineer import FeatureEngineerAgent
from agent_model_trainer import ModelTrainerAgent

class AutoMLPipeline:
    """orchestrates the three-agent pipeline."""
    
    def __init__(self):
        self.logger = AgentLogger()
        self.handoff = Handoff(self.logger)
        Path("outputs").mkdir(exist_ok=True)
        
    def run(self, input_csv: str, target_column: str):
        """execute the full three-agent pipeline."""
        self.logger.log("PIPELINE", "Starting", f"Input: {input_csv}, Target: {target_column}")
        
        # agent 1: data cleaner
        self.logger.log("PIPELINE", "Stage 1", "Initializing Data Cleaner Agent")
        agent1 = DataCleanerAgent(self.logger)
        clean_data_path, report1 = agent1.process(input_csv)
        self.handoff.set_data(clean_data_path, report1, "DataCleaner")
        
        # agent 2: feature engineer
        self.logger.log("PIPELINE", "Stage 2", "Initializing Feature Engineer Agent")
        agent2 = FeatureEngineerAgent(self.logger)
        engineered_data_path, report2 = agent2.process(
            self.handoff.get_data(), 
            self.handoff.get_report(),
            target_column
        )
        self.handoff.set_data(engineered_data_path, report2, "FeatureEngineer")
        
        # agent 3: model trainer
        self.logger.log("PIPELINE", "Stage 3", "Initializing Model Trainer Agent")
        agent3 = ModelTrainerAgent(self.logger)
        final_metrics, report3 = agent3.process(
            self.handoff.get_data(),
            self.handoff.get_report(),
            target_column
        )
        
        # generate final report
        self.generate_final_report(report1, report2, report3, final_metrics)
        
        self.logger.log("PIPELINE", "Complete", "All agents finished successfully")
        self.logger.save()
        
        return final_metrics
    
    def generate_final_report(self, report1, report2, report3, metrics):
        markdown = self.logger.get_markdown_report()
        
        markdown += "\n# Final Summary\n\n"
        markdown += "## Agent 1: Data Cleaner\n"
        markdown += f"- Actions: {report1.get('summary', 'N/A')}\n\n"
        
        markdown += "## Agent 2: Feature Engineer\n"
        markdown += f"- Strategy: {report2.get('summary', 'N/A')}\n\n"
        
        markdown += "## Agent 3: Model Trainer\n"
        markdown += f"- Final Metrics: {metrics}\n\n"
        
        with open("outputs/final_report.md", 'w') as f:
            f.write(markdown)
        
        print("Final Report saved to: outputs/final_report.md")
