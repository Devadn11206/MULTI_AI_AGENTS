import os
import json
import uuid
import datetime
from typing import Dict, Any, Optional
from config.settings import get_settings

class CheckpointManager:
    def __init__(self):
        settings = get_settings()
        self.workspace = os.path.abspath(settings.WORKSPACE_DIR)
        self.checkpoints_dir = os.path.join(self.workspace, "data", "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _get_workflow_dir(self, workflow_id: str) -> str:
        d = os.path.join(self.checkpoints_dir, workflow_id)
        os.makedirs(d, exist_ok=True)
        return d

    def save_checkpoint(self, workflow_id: str, phase: str, agent_name: str, state: Any, status: str = "running"):
        try:
            workflow_dir = self._get_workflow_dir(workflow_id)
            timestamp = datetime.datetime.utcnow().isoformat()
            
            # Serialize state (AgentState is a Pydantic model)
            state_dict = state.model_dump() if hasattr(state, "model_dump") else (state if isinstance(state, dict) else {})
            
            checkpoint = {
                "workflow_id": workflow_id,
                "timestamp": timestamp,
                "phase": phase,
                "current_agent": agent_name,
                "status": status,
                "state": state_dict
            }
            
            # Atomic write for latest checkpoint
            latest_path = os.path.join(workflow_dir, "checkpoint_latest.json")
            temp_path = latest_path + ".tmp"
            
            with open(temp_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
                
            os.replace(temp_path, latest_path)
            
            # Save history checkpoint
            history_path = os.path.join(workflow_dir, f"checkpoint_{int(datetime.datetime.utcnow().timestamp())}.json")
            with open(history_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
                
            print(f"[CheckpointManager] Saved checkpoint for {workflow_id} at {phase}")
        except Exception as e:
            print(f"[CheckpointManager] Error saving checkpoint: {e}")

    def load_latest_checkpoint(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        workflow_dir = self._get_workflow_dir(workflow_id)
        latest_path = os.path.join(workflow_dir, "checkpoint_latest.json")
        
        if not os.path.exists(latest_path):
            return None
            
        try:
            with open(latest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[CheckpointManager] Error loading checkpoint: {e}")
            return None

    def get_recovery_info(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        cp = self.load_latest_checkpoint(workflow_id)
        if not cp:
            return None
        return {
            "workflow_id": cp.get("workflow_id"),
            "phase": cp.get("phase"),
            "status": cp.get("status"),
            "timestamp": cp.get("timestamp")
        }
