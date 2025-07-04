import docker
import asyncio
import logging
import time
import json
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import uuid

from app.core.config import settings, ContainerConfig
from app.models.sandbox import Sandbox, SandboxStatus
from app.security.threat_detector import ThreatDetector

logger = logging.getLogger(__name__)

@dataclass
class SecurityPolicy:
    max_cpu_percent: int = 80
    max_memory_mb: int = 1024
    max_disk_mb: int = 2048
    max_processes: int = 100
    max_open_files: int = 1024
    allowed_ports: List[int] = None
    blocked_commands: List[str] = None
    network_access: bool = True
    internet_access: bool = True

@dataclass
class ContainerInfo:
    container_id: str
    sandbox_id: str
    user_id: str
    status: str
    created_at: datetime
    last_accessed: datetime
    resource_usage: Dict[str, Any]
    policy: SecurityPolicy

class ContainerService:
    """Service for managing Docker containers and sandboxes."""
    
    def __init__(self):
        self.docker_client = None
        self.active_containers: Dict[str, ContainerInfo] = {}
        self.threat_detector = ThreatDetector()
        self.monitoring_task = None
        
    async def initialize(self):
        """Initialize the container service."""
        try:
            self.docker_client = docker.from_env()
            
            # Test Docker connection
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
            
            # Load existing containers
            await self._load_existing_containers()
            
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Stop all containers gracefully
        for container_info in self.active_containers.values():
            try:
                await self._stop_container(container_info.container_id)
            except Exception as e:
                logger.error(f"Error stopping container {container_info.container_id}: {e}")
    
    async def create_sandbox(
        self, 
        sandbox: Sandbox,
        policy: Optional[SecurityPolicy] = None
    ) -> str:
        """Create a new sandbox container."""
        try:
            if policy is None:
                policy = SecurityPolicy(**ContainerConfig.get_security_policy())
            
            container_name = f"sandbox-{sandbox.id}"
            
            # Security configurations
            security_opts = [
                "no-new-privileges:true",
                "seccomp:default"
            ]
            
            # Capabilities - drop all, add only necessary ones
            cap_drop = ["ALL"]
            cap_add = [
                "CHOWN", "DAC_OVERRIDE", "FOWNER", 
                "SETGID", "SETUID", "NET_BIND_SERVICE"
            ]
            
            # Resource constraints
            resource_config = {
                "mem_limit": f"{policy.max_memory_mb}m",
                "memswap_limit": f"{policy.max_memory_mb}m",
                "cpu_quota": int(policy.max_cpu_percent * 1000),
                "cpu_period": 100000,
                "pids_limit": policy.max_processes,
                "ulimits": [
                    docker.types.Ulimit(
                        name='nofile', 
                        soft=policy.max_open_files, 
                        hard=policy.max_open_files
                    ),
                    docker.types.Ulimit(
                        name='nproc', 
                        soft=policy.max_processes, 
                        hard=policy.max_processes
                    )
                ]
            }
            
            # Network configuration
            network_mode = "bridge" if policy.network_access else "none"
            
            # Volume mounts
            volumes = {
                f"sandbox_data_{sandbox.id}": {
                    'bind': '/workspace',
                    'mode': 'rw'
                }
            }
            
            # Environment variables
            environment = {
                'USER_ID': sandbox.user_id,
                'SANDBOX_ID': sandbox.id,
                'TERM': 'xterm-256color',
                'HOME': '/home/sandbox',
                'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                **sandbox.environment_vars
            }
            
            # Create and start container
            container = self.docker_client.containers.run(
                sandbox.image,
                detach=True,
                name=container_name,
                hostname=f"terminal-{sandbox.id[:8]}",
                security_opt=security_opts,
                cap_drop=cap_drop,
                cap_add=cap_add,
                network_mode=network_mode,
                volumes=volumes,
                environment=environment,
                stdin_open=True,
                tty=True,
                working_dir=sandbox.working_directory,
                **resource_config
            )
            
            # Store container info
            container_info = ContainerInfo(
                container_id=container.id,
                sandbox_id=sandbox.id,
                user_id=sandbox.user_id,
                status="running",
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                resource_usage={},
                policy=policy
            )
            
            self.active_containers[sandbox.id] = container_info
            
            # Execute startup script if provided
            if sandbox.startup_script:
                await self._execute_startup_script(container.id, sandbox.startup_script)
            
            logger.info(f"Created sandbox container: {container.id}")
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to create sandbox container: {e}")
            raise
    
    async def start_sandbox(self, sandbox_id: str) -> bool:
        """Start an existing sandbox container."""
        try:
            container_info = self.active_containers.get(sandbox_id)
            if not container_info:
                logger.error(f"Container not found: {sandbox_id}")
                return False
            
            container = self.docker_client.containers.get(container_info.container_id)
            container.start()
            
            container_info.status = "running"
            container_info.last_accessed = datetime.utcnow()
            
            logger.info(f"Started sandbox: {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start sandbox {sandbox_id}: {e}")
            return False
    
    async def stop_sandbox(self, sandbox_id: str) -> bool:
        """Stop a sandbox container."""
        try:
            container_info = self.active_containers.get(sandbox_id)
            if not container_info:
                return False
            
            container = self.docker_client.containers.get(container_info.container_id)
            container.stop(timeout=10)
            
            container_info.status = "stopped"
            
            logger.info(f"Stopped sandbox: {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop sandbox {sandbox_id}: {e}")
            return False
    
    async def pause_sandbox(self, sandbox_id: str) -> bool:
        """Pause a sandbox container."""
        try:
            container_info = self.active_containers.get(sandbox_id)
            if not container_info:
                return False
            
            container = self.docker_client.containers.get(container_info.container_id)
            container.pause()
            
            container_info.status = "paused"
            
            logger.info(f"Paused sandbox: {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause sandbox {sandbox_id}: {e}")
            return False
    
    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a sandbox container and its data."""
        try:
            container_info = self.active_containers.get(sandbox_id)
            if container_info:
                container = self.docker_client.containers.get(container_info.container_id)
                container.stop(timeout=10)
                container.remove()
                
                # Remove from tracking
                del self.active_containers[sandbox_id]
            
            # Remove volume
            try:
                volume = self.docker_client.volumes.get(f"sandbox_data_{sandbox_id}")
                volume.remove()
            except docker.errors.NotFound:
                pass
            
            logger.info(f"Deleted sandbox: {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete sandbox {sandbox_id}: {e}")
            return False
    
    async def execute_command(
        self, 
        sandbox_id: str, 
        command: str,
        user: str = "sandbox"
    ) -> Tuple[int, str, str]:
        """Execute a command in a sandbox container."""
        try:
            # Security check
            if settings.ENABLE_THREAT_DETECTION:
                should_block, threats = self.threat_detector.should_block_command(command)
                if should_block:
                    threat_reasons = [t.reason for t in threats]
                    return 1, "", f"Command blocked: {'; '.join(threat_reasons)}"
            
            container_info = self.active_containers.get(sandbox_id)
            if not container_info:
                return 1, "", "Container not found"
            
            container = self.docker_client.containers.get(container_info.container_id)
            
            # Execute command
            exec_result = container.exec_run(
                command,
                user=user,
                environment={"TERM": "xterm-256color"},
                tty=True,
                stdin=True
            )
            
            # Update access time
            container_info.last_accessed = datetime.utcnow()
            
            return exec_result.exit_code, exec_result.output.decode(), ""
            
        except Exception as e:
            logger.error(f"Failed to execute command in {sandbox_id}: {e}")
            return 1, "", str(e)
    
    async def send_input(self, sandbox_id: str, input_data: str):
        """Send input to a sandbox terminal."""
        try:
            container_info = self.active_containers.get(sandbox_id)
            if not container_info:
                return False
            
            # Update access time
            container_info.last_accessed = datetime.utcnow()
            
            # For real terminal input, we'd need to maintain a persistent shell session
            # This is a simplified implementation
            return True
            
        except Exception as e:
            logger.error(f"Failed to send input to {sandbox_id}: {e}")
            return False
    
    async def resize_terminal(self, sandbox_id: str, cols: int, rows: int):
        """Resize terminal in a sandbox."""
        try:
            container_info = self.active_containers.get(sandbox_id)
            if not container_info:
                return False
            
            container = self.docker_client.containers.get(container_info.container_id)
            container.resize(height=rows, width=cols)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resize terminal in {sandbox_id}: {e}")
            return False
    
    async def get_container_stats(self, sandbox_id: str) -> Dict[str, Any]:
        """Get container resource usage statistics."""
        try:
            container_info = self.active_containers.get(sandbox_id)
            if not container_info:
                return {}
            
            container = self.docker_client.containers.get(container_info.container_id)
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
            
            usage_stats = {
                'cpu_percent': cpu_percent,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'memory_percent': memory_percent,
                'network_rx_bytes': network_rx,
                'network_tx_bytes': network_tx,
                'timestamp': time.time()
            }
            
            # Update stored stats
            container_info.resource_usage = usage_stats
            
            return usage_stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for {sandbox_id}: {e}")
            return {}
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        try:
            stats = {
                'total_containers': len(self.active_containers),
                'running_containers': len([
                    c for c in self.active_containers.values() 
                    if c.status == 'running'
                ]),
                'containers_by_status': {},
                'total_memory_mb': 0,
                'total_cpu_percent': 0,
                'timestamp': time.time()
            }
            
            # Count by status
            for container_info in self.active_containers.values():
                status = container_info.status
                stats['containers_by_status'][status] = \
                    stats['containers_by_status'].get(status, 0) + 1
                
                # Aggregate resource usage
                if container_info.resource_usage:
                    stats['total_memory_mb'] += container_info.resource_usage.get('memory_usage_mb', 0)
                    stats['total_cpu_percent'] += container_info.resource_usage.get('cpu_percent', 0)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    async def start_monitoring(self):
        """Start background monitoring of containers."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background loop for monitoring containers."""
        while True:
            try:
                await self._check_container_health()
                await self._check_resource_violations()
                await self._cleanup_idle_containers()
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _check_container_health(self):
        """Check health of all containers."""
        for sandbox_id, container_info in list(self.active_containers.items()):
            try:
                container = self.docker_client.containers.get(container_info.container_id)
                
                # Update status
                container.reload()
                new_status = container.status
                
                if new_status != container_info.status:
                    container_info.status = new_status
                    logger.info(f"Container {sandbox_id} status changed to {new_status}")
                
                # Update resource usage
                if new_status == 'running':
                    await self.get_container_stats(sandbox_id)
                
            except docker.errors.NotFound:
                # Container was deleted externally
                logger.warning(f"Container {sandbox_id} not found, removing from tracking")
                del self.active_containers[sandbox_id]
            except Exception as e:
                logger.error(f"Health check failed for {sandbox_id}: {e}")
    
    async def _check_resource_violations(self):
        """Check for resource policy violations."""
        for sandbox_id, container_info in self.active_containers.items():
            if container_info.status != 'running':
                continue
            
            policy = container_info.policy
            usage = container_info.resource_usage
            
            if not usage:
                continue
            
            violations = []
            
            # Check CPU usage
            if usage.get('cpu_percent', 0) > policy.max_cpu_percent:
                violations.append(f"CPU usage ({usage['cpu_percent']:.1f}%) exceeds limit ({policy.max_cpu_percent}%)")
            
            # Check memory usage
            if usage.get('memory_usage_mb', 0) > policy.max_memory_mb:
                violations.append(f"Memory usage ({usage['memory_usage_mb']:.1f}MB) exceeds limit ({policy.max_memory_mb}MB)")
            
            # Handle violations
            if violations:
                logger.warning(f"Resource violations in {sandbox_id}: {violations}")
                
                # Pause container if too many violations
                if len(violations) >= 2:
                    await self.pause_sandbox(sandbox_id)
    
    async def _cleanup_idle_containers(self):
        """Clean up idle containers based on policy."""
        current_time = datetime.utcnow()
        
        for sandbox_id, container_info in list(self.active_containers.items()):
            idle_minutes = (current_time - container_info.last_accessed).total_seconds() / 60
            
            # Auto-pause after 30 minutes of inactivity
            if (container_info.status == 'running' and 
                idle_minutes > settings.CONTAINER_TIMEOUT_MINUTES):
                
                logger.info(f"Auto-pausing idle container: {sandbox_id}")
                await self.pause_sandbox(sandbox_id)
    
    async def _execute_startup_script(self, container_id: str, script: str):
        """Execute startup script in container."""
        try:
            container = self.docker_client.containers.get(container_id)
            
            # Write script to file and execute
            script_content = f"#!/bin/bash\n{script}"
            exec_result = container.exec_run(
                f"bash -c 'echo {repr(script_content)} > /tmp/startup.sh && chmod +x /tmp/startup.sh && /tmp/startup.sh'",
                user="sandbox"
            )
            
            if exec_result.exit_code != 0:
                logger.warning(f"Startup script failed: {exec_result.output.decode()}")
            
        except Exception as e:
            logger.error(f"Failed to execute startup script: {e}")
    
    async def _load_existing_containers(self):
        """Load existing containers from Docker."""
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"name": "sandbox-"}
            )
            
            for container in containers:
                # Extract sandbox_id from container name
                if container.name.startswith("sandbox-"):
                    sandbox_id = container.name[8:]  # Remove "sandbox-" prefix
                    
                    # Create container info
                    container_info = ContainerInfo(
                        container_id=container.id,
                        sandbox_id=sandbox_id,
                        user_id="unknown",  # Would need to query database
                        status=container.status,
                        created_at=datetime.utcnow(),
                        last_accessed=datetime.utcnow(),
                        resource_usage={},
                        policy=SecurityPolicy()
                    )
                    
                    self.active_containers[sandbox_id] = container_info
            
            logger.info(f"Loaded {len(self.active_containers)} existing containers")
            
        except Exception as e:
            logger.error(f"Failed to load existing containers: {e}")
    
    async def health_check(self) -> bool:
        """Check if container service is healthy."""
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False
    
    async def cleanup_session(self, session_id: str):
        """Clean up resources for a session."""
        # This would clean up any session-specific resources
        pass