# DevOps & Deployment Guide

## 🚀 Infrastructure Architecture

### Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Load Balancer                        │
│                    (NGINX/CloudFlare)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     │           Frontend Cluster            │
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │   Next.js   │   │   │   Next.js   │   │   Next.js   │   │
│  │  Instance 1 │   │   │  Instance 2 │   │  Instance 3 │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     │           Backend Cluster             │
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │   FastAPI   │   │   │   FastAPI   │   │   FastAPI   │   │
│  │  Instance 1 │   │   │  Instance 2 │   │  Instance 3 │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     │         Container Orchestrator        │
│              Kubernetes Cluster                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                Sandbox Nodes                         │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │  │
│  │  │Container│ │Container│ │Container│ │Container│    │  │
│  │  │   1     │ │   2     │ │   3     │ │   4     │    │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     │            Data Layer                 │
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │ PostgreSQL  │   │   │    Redis    │   │   Vector    │   │
│  │   Cluster   │   │   │   Cluster   │   │     DB      │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┴───────────────────────────────────────┘
```

## 🐳 Docker Configuration

### Production Dockerfiles

```dockerfile
# Dockerfile.frontend
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT 3000
ENV NODE_ENV production

CMD ["node", "server.js"]
```

```dockerfile
# Dockerfile.backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
      - NODE_ENV=development
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/terminalpp
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=development
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=terminalpp
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  monitoring:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

volumes:
  postgres_data:
  redis_data:
```

## 🔧 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [18.x]
        python-version: [3.11]

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_terminalpp
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4

    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install frontend dependencies
      working-directory: ./frontend
      run: |
        npm ci
        npm run build

    - name: Run frontend tests
      working-directory: ./frontend
      run: |
        npm run test:ci
        npm run lint
        npm run type-check

    - name: Install backend dependencies
      working-directory: ./backend
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run backend tests
      working-directory: ./backend
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/test_terminalpp
        REDIS_URL: redis://localhost:6379
      run: |
        python -m pytest tests/ -v --cov=app --cov-report=xml
        python -m black --check .
        python -m isort --check-only .
        python -m flake8 .

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-

    - name: Build and push Docker images
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./Dockerfile.production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: build
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment..."
        # Add staging deployment commands

  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment: production

    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment..."
        # Add production deployment commands
```

### Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: terminal-plus-plus
  labels:
    name: terminal-plus-plus

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: terminal-plus-plus
data:
  NODE_ENV: "production"
  ENVIRONMENT: "production"
  LOG_LEVEL: "info"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: terminal-plus-plus
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  jwt-secret: <base64-encoded-jwt-secret>
  encryption-key: <base64-encoded-encryption-key>

---
# k8s/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: terminal-plus-plus
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: ghcr.io/yourusername/terminal-plus-plus/frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: NODE_ENV
        - name: NEXT_PUBLIC_API_URL
          value: "https://api.terminal-plus-plus.com"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5

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
        image: ghcr.io/yourusername/terminal-plus-plus/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: jwt-secret
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: ENVIRONMENT
        volumeMounts:
        - name: docker-socket
          mountPath: /var/run/docker.sock
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock

---
# k8s/services.yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: terminal-plus-plus
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 3000
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: terminal-plus-plus
spec:
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: terminal-plus-plus-ingress
  namespace: terminal-plus-plus
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  tls:
  - hosts:
    - terminal-plus-plus.com
    - api.terminal-plus-plus.com
    secretName: terminal-plus-plus-tls
  rules:
  - host: terminal-plus-plus.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
  - host: api.terminal-plus-plus.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 80
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'frontend'
    static_configs:
      - targets: ['frontend:3000']
    metrics_path: '/api/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### Application Metrics

```python
# backend/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_SANDBOXES = Gauge(
    'active_sandboxes_total',
    'Number of active sandboxes'
)

SANDBOX_OPERATIONS = Counter(
    'sandbox_operations_total',
    'Total sandbox operations',
    ['operation', 'status']
)

AI_REQUESTS = Counter(
    'ai_requests_total',
    'Total AI requests',
    ['model', 'status']
)

CONTAINER_RESOURCE_USAGE = Gauge(
    'container_resource_usage',
    'Container resource usage',
    ['sandbox_id', 'resource_type']
)

def monitor_requests(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await f(*args, **kwargs)
            status_code = getattr(result, 'status_code', 200)
            REQUEST_COUNT.labels(
                method='POST',  # Extract from request
                endpoint=f.__name__,
                status_code=status_code
            ).inc()
            return result
        except Exception as e:
            REQUEST_COUNT.labels(
                method='POST',
                endpoint=f.__name__,
                status_code=500
            ).inc()
            raise
        finally:
            REQUEST_DURATION.labels(
                method='POST',
                endpoint=f.__name__
            ).observe(time.time() - start_time)
    
    return wrapper

class MetricsCollector:
    def __init__(self):
        self.container_manager = None
        
    def update_sandbox_metrics(self, active_count: int):
        ACTIVE_SANDBOXES.set(active_count)
    
    def record_sandbox_operation(self, operation: str, status: str):
        SANDBOX_OPERATIONS.labels(
            operation=operation,
            status=status
        ).inc()
    
    def record_ai_request(self, model: str, status: str):
        AI_REQUESTS.labels(
            model=model,
            status=status
        ).inc()
    
    def update_container_resources(self, sandbox_id: str, cpu_percent: float, memory_mb: float):
        CONTAINER_RESOURCE_USAGE.labels(
            sandbox_id=sandbox_id,
            resource_type='cpu_percent'
        ).set(cpu_percent)
        
        CONTAINER_RESOURCE_USAGE.labels(
            sandbox_id=sandbox_id,
            resource_type='memory_mb'
        ).set(memory_mb)

# FastAPI endpoint for metrics
from fastapi import FastAPI, Response

@app.get("/metrics")
async def get_metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Terminal++ Monitoring",
    "tags": ["terminal-plus-plus"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Active Sandboxes",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_sandboxes_total",
            "legendFormat": "Active Sandboxes"
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "id": 4,
        "title": "Container Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_resource_usage{resource_type=\"cpu_percent\"}",
            "legendFormat": "CPU {{sandbox_id}}"
          },
          {
            "expr": "container_resource_usage{resource_type=\"memory_mb\"}",
            "legendFormat": "Memory {{sandbox_id}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

## 🔥 Auto-Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: terminal-plus-plus
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: active_connections
      target:
        type: AverageValue
        averageValue: "100"

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend-hpa
  namespace: terminal-plus-plus
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: frontend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Cluster Autoscaler

```yaml
# k8s/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/terminal-plus-plus
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
      serviceAccountName: cluster-autoscaler
```

## 📈 Performance Optimization

### Redis Caching Strategy

```python
# backend/cache/cache_manager.py
import redis
import json
import pickle
from typing import Any, Optional
from functools import wraps
import asyncio
import hashlib

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        try:
            self.redis.setex(
                key, 
                ttl, 
                pickle.dumps(value)
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache."""
        try:
            self.redis.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")

def cache_result(ttl: int = 3600, prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = CacheManager(redis_url="redis://localhost:6379")
            cache_key = cache_manager.cache_key(prefix or func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Usage examples
@cache_result(ttl=1800, prefix="user_sandboxes")
async def get_user_sandboxes(user_id: str):
    # Expensive database query
    pass

@cache_result(ttl=300, prefix="sandbox_stats")
async def get_sandbox_statistics(sandbox_id: str):
    # Resource-intensive stats calculation
    pass
```

### Database Optimization

```sql
-- Database indexes for performance
CREATE INDEX CONCURRENTLY idx_sessions_user_id ON sessions(user_id);
CREATE INDEX CONCURRENTLY idx_sessions_sandbox_id ON sessions(sandbox_id);
CREATE INDEX CONCURRENTLY idx_sessions_expires_at ON sessions(expires_at);

CREATE INDEX CONCURRENTLY idx_command_history_session_id ON command_history(session_id);
CREATE INDEX CONCURRENTLY idx_command_history_executed_at ON command_history(executed_at);

CREATE INDEX CONCURRENTLY idx_sandboxes_user_id ON sandboxes(user_id);
CREATE INDEX CONCURRENTLY idx_sandboxes_status ON sandboxes(status);
CREATE INDEX CONCURRENTLY idx_sandboxes_last_accessed ON sandboxes(last_accessed);

CREATE INDEX CONCURRENTLY idx_file_versions_sandbox_id ON file_versions(sandbox_id);
CREATE INDEX CONCURRENTLY idx_file_versions_file_path ON file_versions(file_path);

-- Partitioning for large tables
CREATE TABLE command_history_y2024m01 PARTITION OF command_history
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE command_history_y2024m02 PARTITION OF command_history
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- Add more partitions as needed

-- Database connection pooling
-- backend/core/database.py
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)
```

## 🚨 Alerting & Incident Response

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
- name: terminal-plus-plus-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"

  - alert: TooManySandboxes
    expr: active_sandboxes_total > 1000
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High number of active sandboxes"
      description: "{{ $value }} sandboxes are currently active"

  - alert: ContainerResourceExhaustion
    expr: max(container_resource_usage{resource_type="cpu_percent"}) > 90
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Container CPU usage critical"
      description: "Container {{ $labels.sandbox_id }} using {{ $value }}% CPU"

  - alert: DatabaseConnectionFailure
    expr: up{job="postgres"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
      description: "Cannot connect to PostgreSQL database"
```

### Incident Response Playbook

```markdown
# Incident Response Playbook

## High Error Rate (5xx responses)

### Immediate Actions (0-5 minutes)
1. Check Grafana dashboard for error patterns
2. Examine recent deployments in last 30 minutes
3. Check container logs: `kubectl logs -n terminal-plus-plus deployment/backend`
4. Verify database connectivity

### Investigation (5-15 minutes)
1. Check specific error types in logs
2. Verify external service dependencies
3. Check resource usage (CPU, memory, disk)
4. Review recent code changes

### Mitigation
- If deployment related: Rollback to previous version
- If resource exhaustion: Scale up pods
- If database issues: Check connection pool, run diagnostics

## High Response Time

### Immediate Actions
1. Check if specific endpoints are affected
2. Monitor database query performance
3. Check Redis cache hit rates
4. Verify container resource usage

### Investigation
1. Identify slow queries in database logs
2. Check cache performance metrics
3. Review recent configuration changes
4. Analyze request patterns

### Mitigation
- Scale up backend pods if CPU bound
- Optimize slow database queries
- Increase cache TTL for frequently accessed data
- Add request rate limiting if needed

## Container Resource Exhaustion

### Immediate Actions
1. Identify affected containers
2. Check if containers can be safely terminated
3. Scale down non-critical sandboxes
4. Alert affected users

### Investigation
1. Review container creation patterns
2. Check for resource leaks
3. Verify auto-scaling configuration
4. Analyze user behavior patterns

### Mitigation
- Implement stricter resource limits
- Auto-terminate idle containers
- Scale cluster if consistently hitting limits
```

## 🔄 Backup & Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

set -e

BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Database backup
echo "Backing up PostgreSQL database..."
pg_dump "$DATABASE_URL" | gzip > "$BACKUP_DIR/database.sql.gz"

# Redis backup
echo "Backing up Redis..."
redis-cli --rdb "$BACKUP_DIR/redis.rdb"

# User data backup (sandbox volumes)
echo "Backing up sandbox data..."
docker run --rm -v sandbox_data:/data -v "$BACKUP_DIR":/backup alpine \
    tar czf /backup/sandbox_data.tar.gz -C /data .

# Upload to cloud storage
echo "Uploading to S3..."
aws s3 sync "$BACKUP_DIR" "s3://terminal-plus-plus-backups/$(date +%Y-%m-%d)/"

# Cleanup old backups (keep last 30 days)
find /backups -type d -mtime +30 -exec rm -rf {} \;

echo "Backup completed successfully"
```

### Disaster Recovery Plan

```yaml
# disaster-recovery/restore-procedure.md
# Disaster Recovery Procedure

## Complete System Failure

### Infrastructure Recovery (RTO: 4 hours)
1. Provision new Kubernetes cluster
2. Deploy monitoring stack (Prometheus, Grafana)
3. Restore database from latest backup
4. Deploy application services
5. Restore Redis cache
6. Verify system functionality

### Data Recovery (RPO: 1 hour)
1. Identify latest valid backup
2. Restore database: `psql < database.sql`
3. Restore sandbox data: `tar xzf sandbox_data.tar.gz`
4. Verify data integrity
5. Resume normal operations

### Communication Plan
1. Notify users via status page
2. Send email updates every 30 minutes
3. Update social media channels
4. Coordinate with support team

## Partial Service Degradation

### Database Failure
- Switch to read replica
- Enable maintenance mode
- Restore primary database
- Resume full operations

### Container Orchestrator Failure
- Scale down to essential services
- Migrate to backup cluster
- Restore full capacity
```

This comprehensive DevOps and deployment guide provides production-ready infrastructure, monitoring, and operational procedures for the Terminal++ platform.