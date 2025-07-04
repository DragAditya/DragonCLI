# Container Orchestration & Security Guide

## 🐳 Container Architecture

### Sandbox Container Design

```dockerfile
# sandbox-images/base/Dockerfile
FROM ubuntu:22.04

# Install system essentials
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tree \
    unzip \
    zip \
    jq \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash -u 1000 sandbox && \
    echo "sandbox:sandbox" | chpasswd && \
    usermod -aG sudo sandbox

# Setup workspace
WORKDIR /workspace
RUN chown sandbox:sandbox /workspace

# Install development tools
RUN pip3 install --no-cache-dir \
    requests \
    flask \
    fastapi \
    uvicorn \
    jupyter

# Copy startup script
COPY startup.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/startup.sh

USER sandbox
EXPOSE 8080

CMD ["/usr/local/bin/startup.sh"]
```

```bash
#!/bin/bash
# sandbox-images/base/startup.sh

# Start SSH daemon for terminal access
sudo service ssh start

# Setup development environment
export PATH="/home/sandbox/.local/bin:$PATH"
export PYTHONPATH="/workspace:$PYTHONPATH"

# Initialize workspace if empty
if [ ! -f "/workspace/.initialized" ]; then
    cat > /workspace/README.md << 'EOF'
# Welcome to Terminal++

This is your persistent development sandbox. Everything in `/workspace` is saved automatically.

## Quick Start
- Run `python3 --version` to check Python
- Run `node --version` to check Node.js
- Type `help` for available commands

Happy coding! 🚀
EOF
    touch /workspace/.initialized
fi

# Start terminal server
exec /bin/bash
```

## 🔒 Security Implementation

### Container Security Manager

```python
# backend/services/container_security.py
import docker
import subprocess
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import psutil

@dataclass
class SecurityPolicy:
    max_cpu_percent: int = 50
    max_memory_mb: int = 512
    max_disk_mb: int = 1024
    max_processes: int = 100
    max_open_files: int = 1024
    allowed_ports: List[int] = None
    blocked_commands: List[str] = None
    network_access: bool = True
    internet_access: bool = True

class ContainerSecurityManager:
    def __init__(self):
        self.client = docker.from_env()
        self.active_containers = {}
        self.security_policies = {}
        
    def create_secure_container(
        self, 
        user_id: str, 
        sandbox_id: str,
        image: str = "terminal-plus-plus/sandbox:latest",
        policy: SecurityPolicy = None
    ) -> str:
        """Create a secure sandbox container with enforced limits."""
        
        if policy is None:
            policy = SecurityPolicy()
            
        # Security configurations
        security_opts = [
            "no-new-privileges:true",
            "seccomp:default"  # Use default seccomp profile
        ]
        
        # Capabilities - drop all, add only necessary ones
        cap_drop = ["ALL"]
        cap_add = [
            "CHOWN",
            "DAC_OVERRIDE", 
            "FOWNER",
            "SETGID",
            "SETUID"
        ]
        
        # Resource constraints
        resource_config = {
            'mem_limit': f"{policy.max_memory_mb}m",
            'memswap_limit': f"{policy.max_memory_mb}m",  # No swap
            'cpu_quota': int(policy.max_cpu_percent * 1000),  # CPU limit
            'cpu_period': 100000,
            'pids_limit': policy.max_processes,
            'ulimits': [
                docker.types.Ulimit(name='nofile', soft=policy.max_open_files, hard=policy.max_open_files),
                docker.types.Ulimit(name='nproc', soft=policy.max_processes, hard=policy.max_processes)
            ]
        }
        
        # Network configuration
        network_mode = "bridge" if policy.network_access else "none"
        
        # Volume mounts with proper permissions
        volumes = {
            f"sandbox_data_{sandbox_id}": {
                'bind': '/workspace',
                'mode': 'rw'
            }
        }
        
        # Environment variables
        environment = {
            'USER_ID': user_id,
            'SANDBOX_ID': sandbox_id,
            'TERM': 'xterm-256color',
            'HOME': '/home/sandbox',
            'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        }
        
        try:
            container = self.client.containers.run(
                image,
                detach=True,
                name=f"sandbox-{sandbox_id}",
                hostname=f"terminal-{sandbox_id[:8]}",
                security_opt=security_opts,
                cap_drop=cap_drop,
                cap_add=cap_add,
                network_mode=network_mode,
                volumes=volumes,
                environment=environment,
                stdin_open=True,
                tty=True,
                **resource_config
            )
            
            # Store container info
            self.active_containers[sandbox_id] = {
                'container_id': container.id,
                'user_id': user_id,
                'created_at': time.time(),
                'policy': policy
            }
            
            self.security_policies[sandbox_id] = policy
            
            # Start monitoring
            self._start_monitoring(sandbox_id, container.id)
            
            return container.id
            
        except Exception as e:
            raise Exception(f"Failed to create secure container: {str(e)}")
    
    def _start_monitoring(self, sandbox_id: str, container_id: str):
        """Start monitoring container for security violations."""
        # This would typically run in a separate thread or process
        pass
    
    def monitor_container_resources(self, container_id: str) -> Dict:
        """Monitor container resource usage."""
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # Parse CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100
                
            # Parse memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100
            
            # Network I/O
            networks = stats.get('networks', {})
            network_rx = sum(net['rx_bytes'] for net in networks.values())
            network_tx = sum(net['tx_bytes'] for net in networks.values())
            
            return {
                'cpu_percent': cpu_percent,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'memory_percent': memory_percent,
                'network_rx_bytes': network_rx,
                'network_tx_bytes': network_tx,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def enforce_security_policy(self, sandbox_id: str) -> Dict:
        """Enforce security policy and handle violations."""
        if sandbox_id not in self.active_containers:
            return {'error': 'Container not found'}
            
        container_info = self.active_containers[sandbox_id]
        policy = self.security_policies[sandbox_id]
        container_id = container_info['container_id']
        
        violations = []
        
        # Check resource usage
        stats = self.monitor_container_resources(container_id)
        
        if stats.get('cpu_percent', 0) > policy.max_cpu_percent:
            violations.append(f"CPU usage ({stats['cpu_percent']:.1f}%) exceeds limit ({policy.max_cpu_percent}%)")
            
        if stats.get('memory_usage_mb', 0) > policy.max_memory_mb:
            violations.append(f"Memory usage ({stats['memory_usage_mb']:.1f}MB) exceeds limit ({policy.max_memory_mb}MB)")
        
        # Check running processes
        try:
            container = self.client.containers.get(container_id)
            processes = container.top()
            process_count = len(processes['Processes'])
            
            if process_count > policy.max_processes:
                violations.append(f"Process count ({process_count}) exceeds limit ({policy.max_processes})")
                
        except Exception as e:
            violations.append(f"Failed to check processes: {str(e)}")
        
        # Handle violations
        if violations:
            self._handle_violations(sandbox_id, violations)
            
        return {
            'violations': violations,
            'stats': stats,
            'timestamp': time.time()
        }
    
    def _handle_violations(self, sandbox_id: str, violations: List[str]):
        """Handle security policy violations."""
        container_info = self.active_containers[sandbox_id]
        
        # Log violations
        print(f"Security violations for sandbox {sandbox_id}: {violations}")
        
        # For severe violations, pause or terminate container
        severe_violations = [v for v in violations if 'exceeds limit' in v]
        
        if len(severe_violations) > 2:
            self.pause_container(sandbox_id)
            
    def pause_container(self, sandbox_id: str) -> bool:
        """Pause a container temporarily."""
        try:
            container_info = self.active_containers[sandbox_id]
            container = self.client.containers.get(container_info['container_id'])
            container.pause()
            return True
        except Exception as e:
            print(f"Failed to pause container {sandbox_id}: {str(e)}")
            return False
    
    def terminate_container(self, sandbox_id: str) -> bool:
        """Safely terminate a container."""
        try:
            container_info = self.active_containers[sandbox_id]
            container = self.client.containers.get(container_info['container_id'])
            
            # Graceful shutdown
            container.stop(timeout=10)
            container.remove()
            
            # Cleanup
            del self.active_containers[sandbox_id]
            del self.security_policies[sandbox_id]
            
            return True
        except Exception as e:
            print(f"Failed to terminate container {sandbox_id}: {str(e)}")
            return False
```

### Advanced Threat Detection

```python
# backend/security/threat_detector.py
import re
import ast
import subprocess
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ThreatDetection:
    command: str
    threat_level: ThreatLevel
    reason: str
    suggested_action: str
    auto_block: bool = False

class AdvancedThreatDetector:
    def __init__(self):
        self.dangerous_patterns = {
            # System destruction
            r'rm\s+-rf\s+/': ThreatLevel.CRITICAL,
            r'dd\s+if=/dev/zero\s+of=/dev/': ThreatLevel.CRITICAL,
            r'mkfs\.': ThreatLevel.CRITICAL,
            r'fdisk\s+/dev/': ThreatLevel.HIGH,
            
            # Network attacks
            r'nmap\s+.*-s[STUN]': ThreatLevel.HIGH,
            r'hydra\s+': ThreatLevel.HIGH,
            r'sqlmap\s+': ThreatLevel.HIGH,
            r'nikto\s+': ThreatLevel.MEDIUM,
            
            # Code injection
            r'eval\s*\(.*\$': ThreatLevel.HIGH,
            r'exec\s*\(.*\$': ThreatLevel.HIGH,
            r'system\s*\(.*\$': ThreatLevel.HIGH,
            
            # Fork bombs and resource exhaustion
            r':\(\)\s*\{\s*:\|:\&\s*\}\s*;:': ThreatLevel.CRITICAL,  # Fork bomb
            r'yes\s+.*\s*>\s*/dev/null\s*&': ThreatLevel.MEDIUM,
            r'while\s+true.*do.*done\s*&': ThreatLevel.MEDIUM,
            
            # Privilege escalation
            r'sudo\s+su\s*-': ThreatLevel.HIGH,
            r'chmod\s+4755': ThreatLevel.HIGH,
            r'chown\s+root:root.*chmod\s+\+s': ThreatLevel.HIGH,
            
            # Data exfiltration
            r'curl\s+.*\|\s*sh': ThreatLevel.HIGH,
            r'wget\s+.*\|\s*sh': ThreatLevel.HIGH,
            r'nc\s+.*-e\s+/bin/sh': ThreatLevel.HIGH,
            
            # Crypto mining
            r'xmrig': ThreatLevel.HIGH,
            r'minerd': ThreatLevel.HIGH,
            r'cpuminer': ThreatLevel.HIGH,
        }
        
        self.suspicious_functions = {
            'eval', 'exec', 'compile', 'open', 'input', '__import__',
            'getattr', 'setattr', 'delattr', 'hasattr'
        }
        
        self.dangerous_modules = {
            'os', 'subprocess', 'sys', 'importlib', 'socket', 'urllib'
        }
    
    def analyze_command(self, command: str) -> List[ThreatDetection]:
        """Analyze a shell command for security threats."""
        threats = []
        
        # Check against pattern database
        for pattern, level in self.dangerous_patterns.items():
            if re.search(pattern, command, re.IGNORECASE):
                threats.append(ThreatDetection(
                    command=command,
                    threat_level=level,
                    reason=f"Matches dangerous pattern: {pattern}",
                    suggested_action="Block command execution",
                    auto_block=(level == ThreatLevel.CRITICAL)
                ))
        
        # Check for suspicious file operations
        if self._has_suspicious_file_ops(command):
            threats.append(ThreatDetection(
                command=command,
                threat_level=ThreatLevel.MEDIUM,
                reason="Suspicious file operations detected",
                suggested_action="Review file access patterns"
            ))
        
        # Check for network activity
        if self._has_network_activity(command):
            threats.append(ThreatDetection(
                command=command,
                threat_level=ThreatLevel.MEDIUM,
                reason="Network activity detected",
                suggested_action="Monitor network connections"
            ))
        
        return threats
    
    def analyze_python_code(self, code: str) -> List[ThreatDetection]:
        """Analyze Python code for security threats."""
        threats = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.suspicious_functions:
                            threats.append(ThreatDetection(
                                command=code,
                                threat_level=ThreatLevel.MEDIUM,
                                reason=f"Use of suspicious function: {func_name}",
                                suggested_action="Review function usage"
                            ))
                
                # Check for dangerous imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_modules:
                            threats.append(ThreatDetection(
                                command=code,
                                threat_level=ThreatLevel.LOW,
                                reason=f"Import of potentially dangerous module: {alias.name}",
                                suggested_action="Review module usage"
                            ))
                
                # Check for exec/eval with user input
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval'] and node.args:
                        threats.append(ThreatDetection(
                            command=code,
                            threat_level=ThreatLevel.HIGH,
                            reason="Dynamic code execution detected",
                            suggested_action="Avoid dynamic code execution",
                            auto_block=True
                        ))
        
        except SyntaxError:
            # If code doesn't parse, it might be obfuscated
            threats.append(ThreatDetection(
                command=code,
                threat_level=ThreatLevel.MEDIUM,
                reason="Code syntax error - possible obfuscation",
                suggested_action="Review code for obfuscation"
            ))
        
        return threats
    
    def _has_suspicious_file_ops(self, command: str) -> bool:
        """Check for suspicious file operations."""
        suspicious_paths = ['/etc/', '/usr/', '/bin/', '/sbin/', '/dev/', '/proc/', '/sys/']
        file_patterns = [r'cat\s+/etc/passwd', r'ls\s+/root', r'find\s+/\s+-name']
        
        for pattern in file_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True
                
        for path in suspicious_paths:
            if path in command and ('rm ' in command or 'chmod ' in command):
                return True
                
        return False
    
    def _has_network_activity(self, command: str) -> bool:
        """Check for network activity."""
        network_commands = ['curl', 'wget', 'nc', 'netcat', 'telnet', 'ftp', 'ssh']
        
        for cmd in network_commands:
            if re.search(rf'\b{cmd}\b', command, re.IGNORECASE):
                return True
                
        # Check for IP addresses or domains
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        domain_pattern = r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        
        return bool(re.search(ip_pattern, command) or re.search(domain_pattern, command))

    def should_block_command(self, command: str) -> Tuple[bool, List[ThreatDetection]]:
        """Determine if a command should be blocked."""
        threats = self.analyze_command(command)
        
        # Block if any critical threats or multiple high threats
        should_block = any(t.auto_block for t in threats) or \
                      len([t for t in threats if t.threat_level == ThreatLevel.HIGH]) >= 2
        
        return should_block, threats
```

## 🎭 Sandbox Templates

### Template System

```python
# backend/services/template_service.py
import json
import os
from typing import Dict, List
from dataclasses import dataclass, asdict

@dataclass
class SandboxTemplate:
    name: str
    description: str
    base_image: str
    packages: List[str]
    environment_vars: Dict[str, str]
    startup_script: str
    files: Dict[str, str]  # path -> content
    ports: List[int]
    category: str
    tags: List[str]
    
class TemplateService:
    def __init__(self):
        self.templates = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in sandbox templates."""
        
        # Python Development Template
        self.templates['python-dev'] = SandboxTemplate(
            name="Python Development",
            description="Full-stack Python development environment",
            base_image="terminal-plus-plus/python:latest",
            packages=[
                "python3-dev", "python3-pip", "python3-venv",
                "postgresql-client", "redis-tools", "git"
            ],
            environment_vars={
                "PYTHONPATH": "/workspace",
                "PIP_USER": "1",
                "VIRTUAL_ENV": "/workspace/venv"
            },
            startup_script="""
#!/bin/bash
cd /workspace
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install fastapi uvicorn requests pytest black isort
fi
source venv/bin/activate
export PS1="(venv) \\u@\\h:\\w\\$ "
""",
            files={
                "requirements.txt": "fastapi==0.104.1\nuvicorn==0.24.0\nrequests==2.31.0\npytest==7.4.3\n",
                "main.py": '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Terminal++!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
                ".gitignore": "__pycache__/\n*.pyc\nvenv/\n.env\n"
            },
            ports=[8000],
            category="development",
            tags=["python", "web", "api", "fastapi"]
        )
        
        # Node.js Template
        self.templates['nodejs-dev'] = SandboxTemplate(
            name="Node.js Development",
            description="Modern Node.js development environment",
            base_image="terminal-plus-plus/nodejs:latest",
            packages=["nodejs", "npm", "yarn", "git"],
            environment_vars={
                "NODE_ENV": "development",
                "NPM_CONFIG_PREFIX": "/workspace/.npm-global"
            },
            startup_script="""
#!/bin/bash
cd /workspace
if [ ! -f "package.json" ]; then
    npm init -y
    npm install express cors helmet morgan
    npm install -D nodemon jest
fi
export PATH="/workspace/.npm-global/bin:$PATH"
""",
            files={
                "package.json": json.dumps({
                    "name": "terminal-plus-plus-app",
                    "version": "1.0.0",
                    "main": "index.js",
                    "scripts": {
                        "start": "node index.js",
                        "dev": "nodemon index.js",
                        "test": "jest"
                    },
                    "dependencies": {
                        "express": "^4.18.2",
                        "cors": "^2.8.5",
                        "helmet": "^7.1.0",
                        "morgan": "^1.10.0"
                    },
                    "devDependencies": {
                        "nodemon": "^3.0.1",
                        "jest": "^29.7.0"
                    }
                }, indent=2),
                "index.js": '''
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());

app.get('/', (req, res) => {
  res.json({ message: 'Hello from Terminal++ Node.js!' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
''',
                ".gitignore": "node_modules/\n.env\nnpm-debug.log*\n"
            },
            ports=[3000],
            category="development", 
            tags=["nodejs", "javascript", "web", "express"]
        )
        
        # Data Science Template
        self.templates['data-science'] = SandboxTemplate(
            name="Data Science",
            description="Python data science and ML environment",
            base_image="terminal-plus-plus/datascience:latest",
            packages=["python3-dev", "python3-pip", "jupyter", "git"],
            environment_vars={
                "JUPYTER_ENABLE_LAB": "yes",
                "PYTHONPATH": "/workspace"
            },
            startup_script="""
#!/bin/bash
cd /workspace
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
fi
source venv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
""",
            files={
                "requirements.txt": "pandas==2.1.3\nnumpy==1.25.2\nmatplotlib==3.8.2\nseaborn==0.13.0\nscikit-learn==1.3.2\njupyter==1.0.0\n",
                "notebook.ipynb": json.dumps({
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": ["# Welcome to Terminal++ Data Science\n\nThis notebook is ready for your data science projects!"]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": ["import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nprint('Data science environment ready!')"]
                        }
                    ],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3"
                        }
                    },
                    "nbformat": 4,
                    "nbformat_minor": 4
                })
            },
            ports=[8888],
            category="data-science",
            tags=["python", "data-science", "jupyter", "ml"]
        )
    
    def get_template(self, template_id: str) -> SandboxTemplate:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, category: str = None) -> List[SandboxTemplate]:
        """List all templates, optionally filtered by category."""
        templates = list(self.templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates
    
    def create_sandbox_from_template(
        self, 
        template_id: str, 
        sandbox_id: str,
        user_customizations: Dict = None
    ) -> Dict:
        """Create a sandbox from a template."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Apply user customizations
        if user_customizations:
            for key, value in user_customizations.items():
                if hasattr(template, key):
                    setattr(template, key, value)
        
        # Create sandbox configuration
        config = {
            'sandbox_id': sandbox_id,
            'template': asdict(template),
            'creation_steps': [
                f"Create container from image: {template.base_image}",
                f"Install packages: {', '.join(template.packages)}",
                f"Set environment variables: {len(template.environment_vars)} vars",
                f"Create files: {len(template.files)} files",
                f"Run startup script",
                f"Expose ports: {template.ports}"
            ]
        }
        
        return config
```

## 🔄 Container Lifecycle Management

```python
# backend/services/lifecycle_manager.py
import asyncio
import time
from typing import Dict, List
from enum import Enum
from dataclasses import dataclass

class ContainerState(Enum):
    CREATING = "creating"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATING = "terminating"

@dataclass
class ContainerInfo:
    sandbox_id: str
    container_id: str
    user_id: str
    state: ContainerState
    created_at: float
    last_accessed: float
    resource_usage: Dict
    auto_pause_time: int = 1800  # 30 minutes
    auto_terminate_time: int = 7200  # 2 hours

class ContainerLifecycleManager:
    def __init__(self):
        self.containers: Dict[str, ContainerInfo] = {}
        self.security_manager = ContainerSecurityManager()
        self.cleanup_task = None
        
    async def start_lifecycle_management(self):
        """Start the lifecycle management background task."""
        self.cleanup_task = asyncio.create_task(self._lifecycle_worker())
    
    async def _lifecycle_worker(self):
        """Background worker for container lifecycle management."""
        while True:
            try:
                await self._check_idle_containers()
                await self._check_resource_violations()
                await self._cleanup_terminated_containers()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Lifecycle worker error: {e}")
                await asyncio.sleep(60)
    
    async def _check_idle_containers(self):
        """Check for idle containers and pause/terminate them."""
        current_time = time.time()
        
        for sandbox_id, info in list(self.containers.items()):
            if info.state != ContainerState.RUNNING:
                continue
                
            idle_time = current_time - info.last_accessed
            
            # Auto-pause after idle time
            if idle_time > info.auto_pause_time and info.state == ContainerState.RUNNING:
                await self._pause_container(sandbox_id, "Auto-paused due to inactivity")
            
            # Auto-terminate after longer idle time
            elif idle_time > info.auto_terminate_time:
                await self._terminate_container(sandbox_id, "Auto-terminated due to extended inactivity")
    
    async def _check_resource_violations(self):
        """Check for resource violations and take action."""
        for sandbox_id, info in list(self.containers.items()):
            if info.state != ContainerState.RUNNING:
                continue
                
            violations = self.security_manager.enforce_security_policy(sandbox_id)
            
            if violations.get('violations'):
                critical_violations = [
                    v for v in violations['violations'] 
                    if 'exceeds limit' in v
                ]
                
                if len(critical_violations) >= 2:
                    await self._pause_container(
                        sandbox_id, 
                        f"Paused due to resource violations: {', '.join(critical_violations)}"
                    )
    
    async def _pause_container(self, sandbox_id: str, reason: str):
        """Pause a container with logging."""
        if sandbox_id in self.containers:
            success = self.security_manager.pause_container(sandbox_id)
            if success:
                self.containers[sandbox_id].state = ContainerState.PAUSED
                print(f"Container {sandbox_id} paused: {reason}")
    
    async def _terminate_container(self, sandbox_id: str, reason: str):
        """Terminate a container with cleanup."""
        if sandbox_id in self.containers:
            success = self.security_manager.terminate_container(sandbox_id)
            if success:
                self.containers[sandbox_id].state = ContainerState.TERMINATING
                print(f"Container {sandbox_id} terminated: {reason}")
    
    async def _cleanup_terminated_containers(self):
        """Remove terminated containers from tracking."""
        to_remove = [
            sandbox_id for sandbox_id, info in self.containers.items()
            if info.state == ContainerState.TERMINATING
        ]
        
        for sandbox_id in to_remove:
            del self.containers[sandbox_id]
    
    def update_access_time(self, sandbox_id: str):
        """Update last accessed time for a container."""
        if sandbox_id in self.containers:
            self.containers[sandbox_id].last_accessed = time.time()
    
    def get_container_stats(self) -> Dict:
        """Get overall container statistics."""
        states = {}
        total_memory = 0
        total_cpu = 0
        
        for info in self.containers.values():
            state = info.state.value
            states[state] = states.get(state, 0) + 1
            
            if info.resource_usage:
                total_memory += info.resource_usage.get('memory_usage_mb', 0)
                total_cpu += info.resource_usage.get('cpu_percent', 0)
        
        return {
            'total_containers': len(self.containers),
            'states': states,
            'total_memory_mb': total_memory,
            'average_cpu_percent': total_cpu / len(self.containers) if self.containers else 0,
            'timestamp': time.time()
        }
```

This container orchestration and security guide provides a comprehensive foundation for secure, scalable sandbox management with advanced threat detection, resource monitoring, and automated lifecycle management.