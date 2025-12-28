from typing import Any, Dict
from logger import AgentLogger
from config import Config
import json
from pathlib import Path

class BaseAgent:    
    def __init__(self, name: str, role: str, logger: AgentLogger):
        self.name = name
        self.role = role
        self.logger = logger
        self.client = Config.get_client()
        self.deployment = Config.AZURE_OPENAI_DEPLOYMENT
        
    def call_llm(self, prompt: str, system_prompt: str) -> str:
        self.logger.log(self.name, "LLM Call", f"Sending prompt to Azure OpenAI...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content
            self.logger.log(self.name, "LLM Response", f"Received response ({len(result)} chars)")
            return result
            
        except Exception as e:
            error_msg = f"Error calling LLM: {str(e)}"
            self.logger.log(self.name, "ERROR", error_msg)
            raise
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        self.logger.log(self.name, f"Tool: {tool_name}", f"Parameters: {kwargs}")
        
        try:
            tool_method = getattr(self, f"_tool_{tool_name}")
            result = tool_method(**kwargs)
            result_str = str(result)[:200] if result is not None else "None"
            self.logger.log(self.name, f"Tool Result: {tool_name}", f"Success: {result_str}")
            return result
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            self.logger.log(self.name, "ERROR", error_msg)
            raise
    
    def save_report(self, report: Dict[str, Any], filename: str):
        report_path = Path(filename)
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, indent=2, fp=f)
        self.logger.log(self.name, "Report Saved", f"Saved to {filename}")


class ToolRegistry:
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, description: str, parameters: Dict[str, str]):
        self.tools[name] = {
            "description": description,
            "parameters": parameters
        }
    
    def get_tool_descriptions(self) -> str:
        descriptions = []
        for name, info in self.tools.items():
            desc = f"\n{name}:\n  {info['description']}\n  Parameters:"
            for param, param_desc in info['parameters'].items():
                desc += f"\n    - {param}: {param_desc}"
            descriptions.append(desc)
        return "\n".join(descriptions)