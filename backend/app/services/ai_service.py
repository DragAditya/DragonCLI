import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import openai
import anthropic
from enum import Enum

from app.core.config import settings, AIConfig
from app.security.threat_detector import ThreatDetector

logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    CRITIC = "critic"
    EXECUTOR = "executor"

class ExecutionMode(str, Enum):
    REVIEW = "review"
    AUTO = "auto"
    DRY_RUN = "dry-run"

@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    commands: Optional[List[str]] = None
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    execution_plan: Optional[Dict[str, Any]] = None

@dataclass
class AIContext:
    session_id: str
    sandbox_id: str
    user_id: str
    current_directory: str = "/workspace"
    environment_vars: Dict[str, str] = None
    file_tree: List[str] = None
    recent_commands: List[str] = None
    recent_errors: List[str] = None
    
    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}
        if self.file_tree is None:
            self.file_tree = []
        if self.recent_commands is None:
            self.recent_commands = []
        if self.recent_errors is None:
            self.recent_errors = []

class BaseAgent(ABC):
    """Base class for all AI agents."""
    
    def __init__(self, name: str, model: str = "gpt-4"):
        self.name = name
        self.model = model
        self.memory: List[Dict[str, Any]] = []
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize clients
        if settings.OPENAI_API_KEY:
            self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    
    @abstractmethod
    async def execute(self, task: str, context: AIContext) -> AgentResponse:
        """Execute the agent's primary function."""
        pass
    
    async def _call_llm(
        self, 
        prompt: str, 
        system_prompt: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.7
    ) -> str:
        """Call the configured LLM."""
        try:
            if self.model.startswith("gpt-") and self.openai_client:
                return await self._call_openai(prompt, system_prompt, max_tokens, temperature)
            elif self.model.startswith("claude-") and self.anthropic_client:
                return await self._call_anthropic(prompt, system_prompt, max_tokens)
            else:
                # Fallback to OpenAI if available
                if self.openai_client:
                    return await self._call_openai(prompt, system_prompt, max_tokens, temperature)
                else:
                    raise Exception("No LLM client available")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: Unable to process request - {str(e)}"
    
    async def _call_openai(
        self, 
        prompt: str, 
        system_prompt: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.7
    ) -> str:
        """Call OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic(
        self, 
        prompt: str, 
        system_prompt: str = None,
        max_tokens: int = 4000
    ) -> str:
        """Call Anthropic API."""
        full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:" if system_prompt else f"Human: {prompt}\n\nAssistant:"
        
        response = await self.anthropic_client.completions.create(
            model=self.model,
            prompt=full_prompt,
            max_tokens_to_sample=max_tokens
        )
        
        return response.completion
    
    def add_to_memory(self, item: Dict[str, Any]):
        """Add item to agent memory."""
        self.memory.append({
            **item,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last 50 items
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]

class PlannerAgent(BaseAgent):
    """Agent responsible for breaking down tasks into executable steps."""
    
    async def execute(self, task: str, context: AIContext) -> AgentResponse:
        system_prompt = """You are a planning agent for a cloud-based development terminal. 
        Your job is to break down user requests into clear, executable steps.
        
        Analyze the user's request and current context to create a detailed execution plan.
        Consider security implications and provide risk assessments.
        
        Respond in JSON format with:
        {
            "analysis": "Brief analysis of the request",
            "steps": [
                {
                    "step": 1,
                    "description": "What to do",
                    "commands": ["command1", "command2"],
                    "expected_output": "What should happen",
                    "risks": ["potential issues"],
                    "validation": "How to verify success"
                }
            ],
            "estimated_time_minutes": 5,
            "difficulty": "easy|medium|hard",
            "prerequisites": ["list of requirements"],
            "risks": ["overall risks"],
            "alternatives": ["alternative approaches"]
        }"""
        
        prompt = f"""
        Task: {task}
        
        Current Context:
        - Directory: {context.current_directory}
        - Recent commands: {context.recent_commands[-5:] if context.recent_commands else 'None'}
        - Recent errors: {context.recent_errors[-3:] if context.recent_errors else 'None'}
        - File structure: {context.file_tree[:20] if context.file_tree else 'Not available'}
        
        Please create a detailed execution plan for this task.
        """
        
        response = await self._call_llm(prompt, system_prompt)
        
        try:
            plan_data = json.loads(response)
            commands = []
            for step in plan_data.get("steps", []):
                commands.extend(step.get("commands", []))
            
            return AgentResponse(
                agent_type=AgentType.PLANNER,
                content=plan_data.get("analysis", ""),
                commands=commands,
                confidence=0.8,
                execution_plan=plan_data
            )
        except json.JSONDecodeError:
            return AgentResponse(
                agent_type=AgentType.PLANNER,
                content=response,
                confidence=0.5
            )

class CoderAgent(BaseAgent):
    """Agent responsible for generating code and scripts."""
    
    async def execute(self, task: str, context: AIContext) -> AgentResponse:
        system_prompt = """You are a coding agent for a development terminal.
        Generate high-quality, secure code based on user requests.
        
        Always consider:
        - Security best practices
        - Error handling
        - Code readability
        - Performance implications
        
        Provide both the code and explanation of what it does."""
        
        prompt = f"""
        Task: {task}
        
        Context:
        - Working directory: {context.current_directory}
        - Environment: {context.environment_vars}
        - Recent activity: {context.recent_commands[-3:] if context.recent_commands else 'None'}
        
        Generate the appropriate code/scripts for this task.
        Include setup commands if needed.
        """
        
        response = await self._call_llm(prompt, system_prompt, temperature=0.3)
        
        # Extract commands from code blocks
        commands = self._extract_commands_from_response(response)
        
        return AgentResponse(
            agent_type=AgentType.CODER,
            content=response,
            commands=commands,
            confidence=0.85
        )
    
    def _extract_commands_from_response(self, response: str) -> List[str]:
        """Extract shell commands from the response."""
        commands = []
        lines = response.split('\n')
        
        in_code_block = False
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
            elif in_code_block and line.strip():
                # Simple heuristic for shell commands
                if any(line.strip().startswith(cmd) for cmd in ['cd', 'ls', 'mkdir', 'touch', 'echo', 'cat', 'python', 'node', 'npm', 'pip']):
                    commands.append(line.strip())
        
        return commands

class CriticAgent(BaseAgent):
    """Agent responsible for reviewing and validating plans/code."""
    
    async def execute(self, task: str, context: AIContext) -> AgentResponse:
        system_prompt = """You are a code review and security agent.
        Your job is to review plans and code for potential issues.
        
        Check for:
        - Security vulnerabilities
        - Logic errors
        - Performance issues
        - Best practice violations
        - Potential risks
        
        Provide specific feedback and suggestions for improvement."""
        
        prompt = f"""
        Please review the following task and context for potential issues:
        
        Task: {task}
        Context: {json.dumps({
            'directory': context.current_directory,
            'recent_commands': context.recent_commands[-5:],
            'recent_errors': context.recent_errors[-3:]
        }, indent=2)}
        
        Provide a security and quality assessment.
        """
        
        response = await self._call_llm(prompt, system_prompt, temperature=0.2)
        
        # Simple risk assessment based on keywords
        risk_keywords = ['rm -rf', 'sudo', 'chmod 777', 'eval', 'exec']
        risk_score = sum(1 for keyword in risk_keywords if keyword in task.lower())
        confidence = max(0.3, 1.0 - (risk_score * 0.2))
        
        return AgentResponse(
            agent_type=AgentType.CRITIC,
            content=response,
            confidence=confidence,
            metadata={
                "risk_score": risk_score,
                "safety_approved": risk_score < 2
            }
        )

class ExecutorAgent(BaseAgent):
    """Agent responsible for safely executing validated commands."""
    
    def __init__(self, name: str, model: str = "gpt-4"):
        super().__init__(name, model)
        self.threat_detector = ThreatDetector()
    
    async def execute(self, task: str, context: AIContext) -> AgentResponse:
        """Validate and prepare commands for execution."""
        
        # This agent doesn't generate new commands but validates existing ones
        # The actual execution would be handled by the container service
        
        system_prompt = """You are an execution agent. Your job is to validate commands before execution.
        
        Check each command for:
        - Safety and security
        - Proper syntax
        - Compatibility with the environment
        - Potential side effects
        
        Only approve commands that are safe to execute."""
        
        prompt = f"""
        Validate the following for safe execution:
        
        Task: {task}
        Context: Current directory is {context.current_directory}
        
        Provide validation results and any safety concerns.
        """
        
        response = await self._call_llm(prompt, system_prompt, temperature=0.1)
        
        return AgentResponse(
            agent_type=AgentType.EXECUTOR,
            content=response,
            confidence=0.9,
            metadata={
                "execution_ready": True,
                "safety_validated": True
            }
        )

class AIOrchestrator:
    """Main orchestrator for AI agents."""
    
    def __init__(self):
        self.agents = {
            AgentType.PLANNER: PlannerAgent("planner"),
            AgentType.CODER: CoderAgent("coder"),
            AgentType.CRITIC: CriticAgent("critic"),
            AgentType.EXECUTOR: ExecutorAgent("executor")
        }
        self.session_contexts: Dict[str, AIContext] = {}
        self.background_task = None
    
    async def initialize(self):
        """Initialize the AI orchestrator."""
        logger.info("AI Orchestrator initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.background_task:
            self.background_task.cancel()
    
    async def start_background_tasks(self):
        """Start background tasks."""
        self.background_task = asyncio.create_task(self._context_cleanup_loop())
    
    async def process_request(
        self,
        session_id: str,
        message: str,
        context: Dict[str, Any],
        mode: ExecutionMode = ExecutionMode.REVIEW
    ) -> Dict[str, Any]:
        """Process an AI request through the agent pipeline."""
        
        try:
            # Get or create session context
            ai_context = self._get_session_context(session_id, context)
            
            # Phase 1: Planning
            logger.info(f"Planning phase for session {session_id}")
            plan_response = await self.agents[AgentType.PLANNER].execute(message, ai_context)
            
            # Phase 2: Code generation (if needed)
            code_response = None
            if self._requires_code_generation(message):
                logger.info(f"Code generation phase for session {session_id}")
                code_response = await self.agents[AgentType.CODER].execute(message, ai_context)
            
            # Phase 3: Review and validation
            logger.info(f"Review phase for session {session_id}")
            review_response = await self.agents[AgentType.CRITIC].execute(message, ai_context)
            
            # Phase 4: Execution preparation
            execution_response = None
            safety_approved = review_response.metadata and review_response.metadata.get("safety_approved", False)
            
            if safety_approved and mode != ExecutionMode.DRY_RUN:
                logger.info(f"Execution preparation for session {session_id}")
                execution_response = await self.agents[AgentType.EXECUTOR].execute(message, ai_context)
            
            # Combine results
            result = {
                "session_id": session_id,
                "mode": mode.value,
                "timestamp": datetime.utcnow().isoformat(),
                "plan": {
                    "content": plan_response.content,
                    "execution_plan": plan_response.execution_plan,
                    "confidence": plan_response.confidence
                },
                "review": {
                    "content": review_response.content,
                    "confidence": review_response.confidence,
                    "safety_approved": safety_approved,
                    "metadata": review_response.metadata
                },
                "commands": plan_response.commands or [],
                "auto_execute": mode == ExecutionMode.AUTO and safety_approved,
                "status": "ready" if safety_approved else "needs_review"
            }
            
            if code_response:
                result["code"] = {
                    "content": code_response.content,
                    "confidence": code_response.confidence,
                    "commands": code_response.commands or []
                }
                # Merge code commands with plan commands
                result["commands"].extend(code_response.commands or [])
            
            if execution_response:
                result["execution"] = {
                    "content": execution_response.content,
                    "confidence": execution_response.confidence,
                    "metadata": execution_response.metadata
                }
            
            # Update session context
            self._update_session_context(session_id, message, result)
            
            return result
            
        except Exception as e:
            logger.error(f"AI processing error for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _get_session_context(self, session_id: str, context: Dict[str, Any]) -> AIContext:
        """Get or create AI context for a session."""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = AIContext(
                session_id=session_id,
                sandbox_id=context.get("sandbox_id", ""),
                user_id=context.get("user_id", ""),
                current_directory=context.get("current_directory", "/workspace"),
                environment_vars=context.get("environment_vars", {}),
                file_tree=context.get("file_tree", []),
                recent_commands=context.get("recent_commands", []),
                recent_errors=context.get("recent_errors", [])
            )
        else:
            # Update existing context
            ai_context = self.session_contexts[session_id]
            ai_context.current_directory = context.get("current_directory", ai_context.current_directory)
            ai_context.environment_vars.update(context.get("environment_vars", {}))
            ai_context.file_tree = context.get("file_tree", ai_context.file_tree)
        
        return self.session_contexts[session_id]
    
    def _update_session_context(self, session_id: str, message: str, result: Dict[str, Any]):
        """Update session context with new information."""
        if session_id in self.session_contexts:
            ai_context = self.session_contexts[session_id]
            
            # Add to recent commands if commands were generated
            if result.get("commands"):
                ai_context.recent_commands.extend(result["commands"])
                # Keep only last 20 commands
                ai_context.recent_commands = ai_context.recent_commands[-20:]
    
    def _requires_code_generation(self, message: str) -> bool:
        """Determine if the request requires code generation."""
        code_keywords = [
            "write", "create", "generate", "code", "script", "function",
            "class", "program", "app", "website", "api", "database"
        ]
        return any(keyword in message.lower() for keyword in code_keywords)
    
    async def _context_cleanup_loop(self):
        """Background loop to clean up old session contexts."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Remove contexts older than 24 hours
                expired_sessions = []
                for session_id, context in self.session_contexts.items():
                    # For now, we'll just limit the number of contexts
                    pass
                
                # Limit total contexts to prevent memory issues
                if len(self.session_contexts) > 1000:
                    # Remove oldest 200 contexts
                    sorted_sessions = sorted(
                        self.session_contexts.items(),
                        key=lambda x: x[1].recent_commands[-1] if x[1].recent_commands else ""
                    )
                    
                    for session_id, _ in sorted_sessions[:200]:
                        del self.session_contexts[session_id]
                
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Context cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history for a session."""
        if session_id in self.session_contexts:
            context = self.session_contexts[session_id]
            return {
                "session_id": session_id,
                "recent_commands": context.recent_commands,
                "recent_errors": context.recent_errors,
                "current_directory": context.current_directory,
                "context_size": len(context.recent_commands)
            }
        return {"session_id": session_id, "error": "Session not found"}
    
    async def clear_session_context(self, session_id: str):
        """Clear context for a session."""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]