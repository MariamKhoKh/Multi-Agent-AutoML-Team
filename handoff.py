import pandas as pd
from pathlib import Path
from typing import Dict, Any
from logger import AgentLogger

class Handoff:
    """manages data and information transfer between agents."""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        self.data_path = None
        self.report = {}
    
    def set_data(self, data_path: str, report: Dict[str, Any], from_agent: str):
        """set the data and report for the next agent."""
        self.data_path = Path(data_path)
        self.report = report
        self.logger.log("HANDOFF", f"Data Transfer from {from_agent}", 
                       f"Data: {data_path}, Report keys: {list(report.keys())}")
    
    def get_data(self) -> pd.DataFrame:
        if self.data_path is None:
            raise ValueError("No data has been set for handoff")
        return pd.read_csv(self.data_path)
    
    def get_report(self) -> Dict[str, Any]:
        return self.report