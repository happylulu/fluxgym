import asyncio
import subprocess
import re
import json
import time
from typing import Optional, Dict, Any, Callable
import torch


class TrainingMonitor:
    """Monitor training progress and GPU usage in real-time"""
    
    def __init__(self):
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        self.gpu_usage = 0
        self.memory_usage = 0
        self.is_training = False
        self.start_time = None
        self.progress_callbacks = []
        
    def add_progress_callback(self, callback: Callable):
        """Add a callback to be called when progress updates"""
        self.progress_callbacks.append(callback)
        
    def parse_training_log(self, line: str) -> Dict[str, Any]:
        """Parse training log line for progress information"""
        updates = {}
        
        # Parse epoch info (e.g., "epoch 1/16")
        epoch_match = re.search(r'epoch\s+(\d+)/(\d+)', line, re.IGNORECASE)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            updates['current_epoch'] = self.current_epoch
            updates['total_epochs'] = total_epochs
            
        # Parse step info (e.g., "steps: 100/1000" or "step 100/1000")
        step_match = re.search(r'step[s]?\s*:?\s*(\d+)/(\d+)', line, re.IGNORECASE)
        if step_match:
            self.current_step = int(step_match.group(1))
            self.total_steps = int(step_match.group(2))
            updates['current_step'] = self.current_step
            updates['total_steps'] = self.total_steps
            updates['progress_percent'] = (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
            
        # Parse loss info
        loss_match = re.search(r'loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
        if loss_match:
            updates['loss'] = float(loss_match.group(1))
            
        # Parse learning rate
        lr_match = re.search(r'lr[:\s=]+([0-9.e-]+)', line, re.IGNORECASE)
        if lr_match:
            updates['learning_rate'] = float(lr_match.group(1))
            
        # Check if training started
        if 'start training' in line.lower() or 'training started' in line.lower():
            self.is_training = True
            self.start_time = time.time()
            updates['training_started'] = True
            
        # Check if training completed
        if 'training complete' in line.lower() or 'saving model' in line.lower():
            updates['training_completed'] = True
            
        return updates
    
    async def monitor_gpu(self) -> Dict[str, float]:
        """Monitor GPU usage using nvidia-smi"""
        try:
            if torch.cuda.is_available():
                # Use PyTorch for more accurate GPU monitoring
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                return {
                    'gpu_usage_percent': gpu_usage,
                    'memory_used_gb': memory_used,
                    'memory_total_gb': memory_total,
                    'device_name': torch.cuda.get_device_name(0)
                }
            else:
                # Fallback to nvidia-smi
                cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,name', '--format=csv,noheader,nounits']
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                
                if proc.returncode == 0:
                    output = stdout.decode().strip()
                    parts = output.split(', ')
                    if len(parts) >= 4:
                        return {
                            'gpu_usage_percent': float(parts[0]),
                            'memory_used_gb': float(parts[1]) / 1024,  # Convert MB to GB
                            'memory_total_gb': float(parts[2]) / 1024,
                            'device_name': parts[3]
                        }
        except Exception as e:
            print(f"Error monitoring GPU: {e}")
            
        return {
            'gpu_usage_percent': 0,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'device_name': 'Unknown'
        }
    
    async def run_training_with_monitoring(self, command: str, cwd: str, env: dict) -> asyncio.subprocess.Process:
        """Run training command with real-time monitoring"""
        # Use stdbuf to ensure line buffering
        if 'bash' in command:
            # Wrap the bash command with stdbuf
            command = f"stdbuf -oL -eL {command}"
            
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env
        )
        
        return proc
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
            'elapsed_time': elapsed_time,
            'gpu_usage': self.gpu_usage,
            'memory_usage': self.memory_usage
        }
    
    def notify_progress(self, updates: Dict[str, Any]):
        """Notify all callbacks of progress updates"""
        for callback in self.progress_callbacks:
            try:
                callback(updates)
            except Exception as e:
                print(f"Error in progress callback: {e}")