# Terminal++ AI Dev Shell - Technical Blueprint

## 🎯 Executive Summary

A cloud-native, browser-based terminal that combines secure containerized environments with multi-agent AI orchestration, providing developers with an intelligent, persistent, and collaborative development platform.

## 🏗️ System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React/Next.js)                │
├─────────────────────────────────────────────────────────────┤
│                     WebSocket Gateway                       │
├─────────────────────────────────────────────────────────────┤
│  API Gateway (FastAPI/Node.js) │  AI Orchestrator Service  │
├─────────────────────────────────────────────────────────────┤
│     Container Orchestrator      │     Memory & Context     │
│        (Docker/K8s)            │        Service           │
├─────────────────────────────────────────────────────────────┤
│  Persistent Storage  │  Vector DB  │  Session Management  │
│   (PostgreSQL)      │ (Pinecone)  │     (Redis)          │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- React 18 + Next.js 14 (App Router)
- TypeScript
- Tailwind CSS + Shadcn/UI
- Xterm.js for terminal emulation
- Socket.io-client for real-time communication
- Monaco Editor for code editing

**Backend:**
- FastAPI (Python) or Node.js with Express
- PostgreSQL with Prisma/SQLAlchemy
- Redis for session management
- Docker for containerization
- Kubernetes for orchestration (optional)

**AI/ML:**
- OpenAI GPT-4/Claude API integration
- Langchain for agent orchestration
- Pinecone/Weaviate for vector storage
- Sentence Transformers for embeddings

**Infrastructure:**
- AWS/GCP/Azure for cloud deployment
- Fly.io or Railway for simplified deployment
- NGINX for load balancing
- Let's Encrypt for SSL certificates

## 🗄️ Database Schema

### Core Tables

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    tier VARCHAR(20) DEFAULT 'free', -- free, pro, enterprise
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Sandboxes table
CREATE TABLE sandboxes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    container_id VARCHAR(255),
    status VARCHAR(20) DEFAULT 'stopped', -- running, stopped, suspended
    cpu_limit INTEGER DEFAULT 1000, -- millicores
    memory_limit INTEGER DEFAULT 512, -- MB
    disk_limit INTEGER DEFAULT 1024, -- MB
    environment JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sandbox_id UUID REFERENCES sandboxes(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT NOW()
);

-- Command history
CREATE TABLE command_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    command TEXT NOT NULL,
    output TEXT,
    exit_code INTEGER,
    executed_at TIMESTAMP DEFAULT NOW(),
    execution_time_ms INTEGER
);

-- AI conversations
CREATE TABLE ai_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- user, assistant, system
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- File versions
CREATE TABLE file_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sandbox_id UUID REFERENCES sandboxes(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    content TEXT,
    size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- Snapshots
CREATE TABLE snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sandbox_id UUID REFERENCES sandboxes(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    snapshot_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- Secrets vault
CREATE TABLE secrets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    sandbox_id UUID REFERENCES sandboxes(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    encrypted_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, sandbox_id, key_name)
);

-- Collaboration
CREATE TABLE collaborations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sandbox_id UUID REFERENCES sandboxes(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    invited_by UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- viewer, editor, owner
    status VARCHAR(20) DEFAULT 'pending', -- pending, active, revoked
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);
```

## 🚀 API Design

### Authentication Endpoints

```python
# auth.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import bcrypt
import jwt

router = APIRouter(prefix="/auth")

class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    username: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    # Verify credentials
    # Return JWT token
    pass

@router.post("/signup")
async def signup(request: SignupRequest):
    # Create user account
    # Send verification email
    pass

@router.post("/logout")
async def logout(token: str = Depends(get_current_user)):
    # Invalidate token
    pass
```

### Sandbox Management

```python
# sandboxes.py
from fastapi import APIRouter, Depends
from typing import List, Optional

router = APIRouter(prefix="/sandboxes")

@router.get("/")
async def list_sandboxes(user: User = Depends(get_current_user)) -> List[Sandbox]:
    # Return user's sandboxes
    pass

@router.post("/")
async def create_sandbox(
    name: str,
    template: Optional[str] = None,
    user: User = Depends(get_current_user)
) -> Sandbox:
    # Create new sandbox container
    # Initialize with template if provided
    pass

@router.get("/{sandbox_id}")
async def get_sandbox(sandbox_id: str, user: User = Depends(get_current_user)):
    # Return sandbox details
    pass

@router.post("/{sandbox_id}/start")
async def start_sandbox(sandbox_id: str, user: User = Depends(get_current_user)):
    # Start container
    # Return connection details
    pass

@router.post("/{sandbox_id}/stop")
async def stop_sandbox(sandbox_id: str, user: User = Depends(get_current_user)):
    # Stop container gracefully
    pass

@router.delete("/{sandbox_id}")
async def delete_sandbox(sandbox_id: str, user: User = Depends(get_current_user)):
    # Delete container and all data
    pass
```

### AI Orchestrator

```python
# ai.py
from fastapi import APIRouter, WebSocket
from langchain.agents import AgentExecutor
from langchain.tools import Tool

router = APIRouter(prefix="/ai")

class AIOrchestrator:
    def __init__(self):
        self.agents = {}
        self.memory = {}
    
    async def process_request(self, session_id: str, message: str, mode: str):
        # Route to appropriate agent
        # Execute command if in auto mode
        # Return response
        pass

@router.websocket("/chat/{session_id}")
async def ai_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    orchestrator = AIOrchestrator()
    
    while True:
        data = await websocket.receive_json()
        response = await orchestrator.process_request(
            session_id, 
            data['message'], 
            data.get('mode', 'review')
        )
        await websocket.send_json(response)
```

### File System Operations

```python
# filesystem.py
from fastapi import APIRouter, UploadFile, File
import os
import aiofiles

router = APIRouter(prefix="/fs")

@router.get("/{sandbox_id}/tree")
async def get_file_tree(sandbox_id: str, path: str = "/"):
    # Return directory structure
    pass

@router.get("/{sandbox_id}/file")
async def read_file(sandbox_id: str, path: str):
    # Return file contents
    pass

@router.post("/{sandbox_id}/file")
async def write_file(sandbox_id: str, path: str, content: str):
    # Write file contents
    # Create version entry
    pass

@router.delete("/{sandbox_id}/file")
async def delete_file(sandbox_id: str, path: str):
    # Delete file
    # Log operation
    pass

@router.post("/{sandbox_id}/upload")
async def upload_file(sandbox_id: str, file: UploadFile = File(...)):
    # Handle file upload
    pass
```

## 🧠 AI Agent Architecture

### Multi-Agent System

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgent(ABC):
    def __init__(self, name: str, model: str = "gpt-4"):
        self.name = name
        self.model = model
        self.memory = []
        
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def add_to_memory(self, item: Dict[str, Any]):
        self.memory.append(item)

# agents/planner.py
class PlannerAgent(BaseAgent):
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Break down task into steps
        # Validate feasibility
        # Return execution plan
        prompt = f"""
        Task: {task}
        Context: {context}
        
        Break this down into executable steps with validation checks.
        """
        # Call LLM and parse response
        return {"steps": [], "estimated_time": 0, "risks": []}

# agents/coder.py
class CoderAgent(BaseAgent):
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Generate code based on requirements
        # Consider existing codebase
        # Apply best practices
        pass

# agents/critic.py
class CriticAgent(BaseAgent):
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Review code/plans for issues
        # Suggest improvements
        # Check security implications
        pass

# agents/executor.py
class ExecutorAgent(BaseAgent):
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Execute validated commands
        # Monitor progress
        # Handle errors
        pass
```

### Agent Orchestration

```python
# orchestrator.py
from typing import List, Dict, Any
import asyncio

class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            'planner': PlannerAgent('planner'),
            'coder': CoderAgent('coder'),
            'critic': CriticAgent('critic'),
            'executor': ExecutorAgent('executor')
        }
        
    async def process_request(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Planning phase
        plan = await self.agents['planner'].execute(request, context)
        
        # 2. Code generation (if needed)
        if plan.get('requires_code'):
            code = await self.agents['coder'].execute(request, {**context, 'plan': plan})
            plan['code'] = code
            
        # 3. Review phase
        review = await self.agents['critic'].execute(request, {**context, 'plan': plan})
        
        # 4. Execution (if approved)
        if review.get('approved') and context.get('auto_execute', False):
            result = await self.agents['executor'].execute(request, {**context, 'plan': plan})
            return result
            
        return {'plan': plan, 'review': review, 'status': 'pending_approval'}
```

## 🔐 Security Implementation

### Container Security

```python
# security/container.py
import docker
from typing import Dict, List

class SecureContainer:
    def __init__(self, user_id: str, sandbox_id: str):
        self.user_id = user_id
        self.sandbox_id = sandbox_id
        self.client = docker.from_env()
        
    def create_container(self, image: str = "ubuntu:20.04") -> str:
        # Security constraints
        security_opts = [
            "no-new-privileges:true",
            "seccomp:unconfined"  # Customize as needed
        ]
        
        # Resource limits
        container = self.client.containers.run(
            image,
            detach=True,
            mem_limit="512m",
            cpu_quota=50000,  # 50% of one CPU
            network_mode="bridge",
            security_opt=security_opts,
            cap_drop=["ALL"],
            cap_add=["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"],
            read_only=False,  # Allow writes to specific directories
            tmpfs={'/tmp': 'size=100m'},
            name=f"sandbox-{self.sandbox_id}"
        )
        
        return container.id

# security/threat_detection.py
class ThreatDetector:
    def __init__(self):
        self.dangerous_commands = [
            r'rm\s+-rf\s+/',
            r'dd\s+if=.*of=/dev/',
            r'mkfs\.',
            r':(){ :|:& };:',  # Fork bomb
            r'wget.*\|\s*sh',
            r'curl.*\|\s*sh'
        ]
        
    def is_command_safe(self, command: str) -> tuple[bool, str]:
        import re
        for pattern in self.dangerous_commands:
            if re.search(pattern, command):
                return False, f"Potentially dangerous command detected: {pattern}"
        return True, "Command appears safe"
        
    def sanitize_environment(self, env_vars: Dict[str, str]) -> Dict[str, str]:
        # Remove sensitive environment variables
        sensitive_keys = ['AWS_SECRET', 'PRIVATE_KEY', 'PASSWORD']
        return {k: v for k, v in env_vars.items() 
                if not any(sensitive in k.upper() for sensitive in sensitive_keys)}
```

### Secrets Management

```python
# security/secrets.py
from cryptography.fernet import Fernet
import os
import base64

class SecretsVault:
    def __init__(self):
        self.key = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.key)
        
    def encrypt_secret(self, value: str) -> str:
        return self.cipher.encrypt(value.encode()).decode()
        
    def decrypt_secret(self, encrypted_value: str) -> str:
        return self.cipher.decrypt(encrypted_value.encode()).decode()
        
    async def store_secret(self, user_id: str, sandbox_id: str, key: str, value: str):
        encrypted_value = self.encrypt_secret(value)
        # Store in database
        pass
        
    async def retrieve_secret(self, user_id: str, sandbox_id: str, key: str) -> str:
        # Retrieve from database
        # Decrypt and return
        pass
```

## 📁 Project Structure

```
terminal-plus-plus/
├── frontend/                   # React/Next.js frontend
│   ├── components/
│   │   ├── Terminal/
│   │   │   ├── TerminalComponent.tsx
│   │   │   ├── CommandHistory.tsx
│   │   │   └── AIAssistant.tsx
│   │   ├── FileExplorer/
│   │   │   ├── FileTree.tsx
│   │   │   ├── FileEditor.tsx
│   │   │   └── DiffViewer.tsx
│   │   ├── Sandbox/
│   │   │   ├── SandboxList.tsx
│   │   │   ├── SandboxCreator.tsx
│   │   │   └── ResourceMonitor.tsx
│   │   └── UI/
│   │       ├── Layout.tsx
│   │       ├── Navigation.tsx
│   │       └── Modals.tsx
│   ├── pages/
│   │   ├── api/
│   │   ├── dashboard/
│   │   ├── sandbox/
│   │   └── settings/
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useTerminal.ts
│   │   └── useAI.ts
│   ├── utils/
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   └── constants.ts
│   └── styles/
│       └── globals.css
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   ├── sandboxes.py
│   │   │   ├── filesystem.py
│   │   │   ├── ai.py
│   │   │   └── websocket.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── security.py
│   │   │   └── dependencies.py
│   │   ├── models/
│   │   │   ├── user.py
│   │   │   ├── sandbox.py
│   │   │   ├── session.py
│   │   │   └── file.py
│   │   ├── services/
│   │   │   ├── container_service.py
│   │   │   ├── ai_service.py
│   │   │   ├── github_service.py
│   │   │   └── storage_service.py
│   │   ├── agents/
│   │   │   ├── base.py
│   │   │   ├── planner.py
│   │   │   ├── coder.py
│   │   │   ├── critic.py
│   │   │   └── executor.py
│   │   └── security/
│   │       ├── container.py
│   │       ├── secrets.py
│   │       └── threat_detection.py
│   ├── migrations/
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── infrastructure/             # Infrastructure as code
│   ├── docker/
│   │   ├── Dockerfile.frontend
│   │   ├── Dockerfile.backend
│   │   ├── Dockerfile.sandbox
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployments/
│   │   ├── services/
│   │   └── ingress/
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── helm/
│       └── terminal-plus-plus/
├── sandbox-images/             # Custom sandbox images
│   ├── base/
│   │   └── Dockerfile
│   ├── nodejs/
│   │   └── Dockerfile
│   ├── python/
│   │   └── Dockerfile
│   └── golang/
│       └── Dockerfile
├── docs/                       # Documentation
│   ├── api.md
│   ├── deployment.md
│   ├── security.md
│   └── contributing.md
├── scripts/                    # Utility scripts
│   ├── setup.sh
│   ├── deploy.sh
│   └── backup.sh
└── README.md
```

## 🚢 Deployment Strategy

### Docker Configuration

```dockerfile
# Dockerfile.backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/terminalpp
      - REDIS_URL=redis://redis:6379
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=terminalpp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: terminal-plus-plus

---
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: terminal-plus-plus
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: terminal-plus-plus/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

---
# k8s/backend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: terminal-plus-plus
spec:
  selector:
    app: backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

## 📋 Implementation Roadmap

### Phase 1: MVP (4-6 weeks)
- [ ] Basic authentication system
- [ ] Simple sandbox creation with Docker
- [ ] Web-based terminal interface
- [ ] Basic AI chat integration
- [ ] File system CRUD operations
- [ ] Simple deployment setup

### Phase 2: AI Enhancement (6-8 weeks)
- [ ] Multi-agent system implementation
- [ ] Command execution automation
- [ ] Memory and context management
- [ ] Threat detection system
- [ ] Advanced AI interactions

### Phase 3: Version Control (4-6 weeks)
- [ ] GitHub integration
- [ ] File versioning system
- [ ] Snapshot and restore functionality
- [ ] Visual diff viewer
- [ ] Git operations through UI

### Phase 4: Collaboration (6-8 weeks)
- [ ] Real-time collaboration
- [ ] Shared sandboxes
- [ ] Permission management
- [ ] Activity logging and audit trails

### Phase 5: Advanced Features (8-10 weeks)
- [ ] Plugin system
- [ ] Multiple LLM support
- [ ] Advanced monitoring
- [ ] Mobile optimization
- [ ] Offline mode

### Phase 6: Production Ready (4-6 weeks)
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] CI/CD pipeline

## 🔧 Development Setup

```bash
# setup.sh
#!/bin/bash

# Clone repository
git clone https://github.com/yourusername/terminal-plus-plus.git
cd terminal-plus-plus

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup database
docker run -d --name postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=terminalpp \
  -p 5432:5432 \
  postgres:15

# Run migrations
alembic upgrade head

# Start backend
uvicorn app.main:app --reload &

# Setup frontend
cd ../frontend
npm install
npm run dev &

echo "Development environment ready!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:8000"
```

## 🎯 Success Metrics

- **User Engagement**: Daily/Monthly Active Users
- **Performance**: Container startup time < 5 seconds
- **Reliability**: 99.9% uptime SLA
- **Security**: Zero critical vulnerabilities
- **Scalability**: Support 10,000+ concurrent users
- **Cost Efficiency**: < $0.10 per user per hour

This blueprint provides a comprehensive foundation for building the Terminal++ AI Dev Shell. Each component is designed to be modular, scalable, and secure, allowing for iterative development and feature expansion.