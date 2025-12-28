import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from pathlib import Path
from logger import AgentLogger
from agent_model_trainer import ModelTrainerAgent
from config import Config
import json

def test_agent3():
    print("Testing Agent 3: Model Trainer")
    
    try:
        Config.validate()
        print("Azure OpenAI configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    engineered_data_path = "outputs/engineered_data.csv"
    
    if not Path(engineered_data_path).exists():
        print(f"Error: {engineered_data_path} not found")
        print("  Please run test_agent2.py first to generate engineered data")
        return
    
    df = pd.read_csv(engineered_data_path)
    print(f"Loaded engineered data from Agent 2")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {list(df.columns)}\n")
    
    previous_report = {
        "agent": "FeatureEngineer",
        "summary": "Created 2 interaction features and encoded categorical variables"
    }
    
    logger = AgentLogger("outputs/test_agent3.log")
    agent = ModelTrainerAgent(logger)
    
    print("Agent 3 is now training models with feedback loop...")
    
    try:
        final_metrics, report = agent.process(df, previous_report, "target")
        
        print("AGENT 3 RESULTS")
        
        print(f"\nTraining complete!")
        print(f"Report saved to: outputs/model_trainer_report.json")
        print(f"Final code saved to: outputs/final_model_code.py")
        
        print(f"\nTotal Iterations: {report['total_iterations']}")
        
        print("\nTraining History:")
        for entry in report['training_history']:
            print(f"  Iteration {entry['iteration']}:")
            for metric, value in entry['metrics'].items():
                print(f"    - {metric}: {value:.4f}")
        
        print("\nFinal Metrics:")
        for metric, value in final_metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        print(f"\nSummary: {report['summary']}")
        
        print("\nFinal Model Code (first 500 chars):")
        print(report['final_code'][:500] + "...")
        
        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")
        
        print("FEEDBACK LOOP VERIFICATION")
        if report['total_iterations'] > 1:
            print(f"Agent made {report['total_iterations']} attempts")
            print("Feedback loop worked: Agent iterated to improve performance")
        else:
            print("Agent achieved satisfactory performance in first attempt")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent3()