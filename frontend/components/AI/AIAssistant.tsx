'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Bot, 
  Send, 
  Loader2, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Play,
  Pause,
  Settings,
  Minimize2,
  Maximize2,
  RefreshCw
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface AIMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  agentType?: 'planner' | 'coder' | 'critic' | 'executor';
  commands?: string[];
  confidence?: number;
  metadata?: Record<string, any>;
}

interface AIAssistantProps {
  sandboxId: string;
  sessionId: string;
  onCommandSuggest?: (commands: string[]) => void;
  onExecuteCommands?: (commands: string[]) => void;
  className?: string;
  isMinimized?: boolean;
  onToggleMinimize?: () => void;
}

interface AIResponse {
  session_id: string;
  mode: 'review' | 'auto' | 'dry-run';
  plan: {
    content: string;
    execution_plan: any;
    confidence: number;
  };
  review: {
    content: string;
    confidence: number;
    safety_approved: boolean;
    metadata: any;
  };
  commands: string[];
  auto_execute: boolean;
  status: 'ready' | 'needs_review' | 'error';
  code?: {
    content: string;
    confidence: number;
    commands: string[];
  };
  execution?: {
    content: string;
    confidence: number;
    metadata: any;
  };
}

const ExecutionMode = {
  REVIEW: 'review' as const,
  AUTO: 'auto' as const,
  DRY_RUN: 'dry-run' as const
};

export const AIAssistant: React.FC<AIAssistantProps> = ({
  sandboxId,
  sessionId,
  onCommandSuggest,
  onExecuteCommands,
  className = '',
  isMinimized = false,
  onToggleMinimize
}) => {
  const [messages, setMessages] = useState<AIMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [executionMode, setExecutionMode] = useState<'review' | 'auto' | 'dry-run'>('review');
  const [isConnected, setIsConnected] = useState(false);
  const [currentContext, setCurrentContext] = useState<Record<string, any>>({});
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize with welcome message
  useEffect(() => {
    const welcomeMessage: AIMessage = {
      id: 'welcome',
      type: 'system',
      content: `🤖 **AI Assistant Ready**

I'm your AI development buddy! I can help you with:
- 📋 Planning and breaking down tasks
- 💻 Writing code and scripts  
- 🔍 Reviewing code for security and best practices
- ⚡ Executing commands safely
- 🛠️ Debugging and troubleshooting

**Execution Modes:**
- **Review Mode**: I'll suggest commands for your approval
- **Auto Mode**: I'll execute safe commands automatically  
- **Dry Run**: I'll show what would happen without executing

Try asking me something like "Create a Python web server" or "Help me set up a React app"!`,
      timestamp: new Date(),
      agentType: 'planner'
    };
    
    setMessages([welcomeMessage]);
  }, []);

  const sendMessage = useCallback(async (message: string) => {
    if (!message.trim() || isLoading) return;

    const userMessage: AIMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: message,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Send to AI API
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: message,
          context: {
            sandbox_id: sandboxId,
            user_id: 'current-user', // TODO: Get from auth
            current_directory: currentContext.currentDirectory || '/workspace',
            environment_vars: currentContext.environmentVars || {},
            file_tree: currentContext.fileTree || [],
            recent_commands: currentContext.recentCommands || [],
            recent_errors: currentContext.recentErrors || []
          },
          mode: executionMode
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const aiResponse: AIResponse = await response.json();
      
      // Create assistant messages for each agent response
      const assistantMessages: AIMessage[] = [];

      // Planning response
      if (aiResponse.plan) {
        assistantMessages.push({
          id: `planner-${Date.now()}`,
          type: 'assistant',
          content: `## 📋 Planning Phase\n\n${aiResponse.plan.content}`,
          timestamp: new Date(),
          agentType: 'planner',
          confidence: aiResponse.plan.confidence,
          metadata: { execution_plan: aiResponse.plan.execution_plan }
        });
      }

      // Code generation response
      if (aiResponse.code) {
        assistantMessages.push({
          id: `coder-${Date.now()}`,
          type: 'assistant',
          content: `## 💻 Code Generation\n\n${aiResponse.code.content}`,
          timestamp: new Date(),
          agentType: 'coder',
          confidence: aiResponse.code.confidence,
          commands: aiResponse.code.commands
        });
      }

      // Review response
      if (aiResponse.review) {
        const reviewIcon = aiResponse.review.safety_approved ? '✅' : '⚠️';
        const reviewStatus = aiResponse.review.safety_approved ? 'Approved' : 'Needs Review';
        
        assistantMessages.push({
          id: `critic-${Date.now()}`,
          type: 'assistant',
          content: `## ${reviewIcon} Security Review - ${reviewStatus}\n\n${aiResponse.review.content}`,
          timestamp: new Date(),
          agentType: 'critic',
          confidence: aiResponse.review.confidence,
          metadata: aiResponse.review.metadata
        });
      }

      // Execution response
      if (aiResponse.execution) {
        assistantMessages.push({
          id: `executor-${Date.now()}`,
          type: 'assistant',
          content: `## ⚡ Execution Plan\n\n${aiResponse.execution.content}`,
          timestamp: new Date(),
          agentType: 'executor',
          confidence: aiResponse.execution.confidence,
          metadata: aiResponse.execution.metadata
        });
      }

      // Commands summary
      if (aiResponse.commands && aiResponse.commands.length > 0) {
        const commandsMessage: AIMessage = {
          id: `commands-${Date.now()}`,
          type: 'assistant',
          content: `## 🔧 Suggested Commands\n\n${aiResponse.commands.map(cmd => `\`\`\`bash\n${cmd}\n\`\`\``).join('\n\n')}`,
          timestamp: new Date(),
          commands: aiResponse.commands,
          metadata: { 
            status: aiResponse.status,
            auto_execute: aiResponse.auto_execute,
            safety_approved: aiResponse.review?.safety_approved 
          }
        };
        assistantMessages.push(commandsMessage);

        // Notify parent component about commands
        onCommandSuggest?.(aiResponse.commands);

        // Auto-execute if approved and in auto mode
        if (aiResponse.auto_execute && aiResponse.review?.safety_approved) {
          onExecuteCommands?.(aiResponse.commands);
        }
      }

      setMessages(prev => [...prev, ...assistantMessages]);

    } catch (error) {
      console.error('AI request failed:', error);
      
      const errorMessage: AIMessage = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: `❌ **Error**: Failed to process your request. Please try again.`,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [
    sessionId, 
    sandboxId, 
    executionMode, 
    currentContext, 
    onCommandSuggest, 
    onExecuteCommands,
    isLoading
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleExecuteCommands = (commands: string[]) => {
    onExecuteCommands?.(commands);
    
    const executionMessage: AIMessage = {
      id: `execution-${Date.now()}`,
      type: 'system',
      content: `🚀 **Executing Commands:**\n\n${commands.map(cmd => `\`${cmd}\``).join('\n')}`,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, executionMessage]);
  };

  const getAgentColor = (agentType?: string) => {
    switch (agentType) {
      case 'planner': return 'text-blue-400';
      case 'coder': return 'text-green-400';
      case 'critic': return 'text-yellow-400';
      case 'executor': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  };

  const getAgentIcon = (agentType?: string) => {
    switch (agentType) {
      case 'planner': return '📋';
      case 'coder': return '💻';
      case 'critic': return '🔍';
      case 'executor': return '⚡';
      default: return '🤖';
    }
  };

  if (isMinimized) {
    return (
      <div className={`ai-assistant-minimized ${className}`}>
        <button
          onClick={onToggleMinimize}
          className="minimized-button"
        >
          <Bot size={20} />
          <span>AI Assistant</span>
          {isLoading && <Loader2 size={16} className="animate-spin" />}
        </button>
        
        <style jsx>{`
          .ai-assistant-minimized {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 100;
          }
          
          .minimized-button {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            background: var(--ai-bg, #1f2937);
            color: var(--ai-text, #ffffff);
            border: 1px solid var(--ai-border, #374151);
            border-radius: 24px;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
          }
          
          .minimized-button:hover {
            background: var(--ai-hover, #374151);
            transform: translateY(-2px);
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className={`ai-assistant ${className}`}>
      {/* Header */}
      <div className="ai-header">
        <div className="ai-title">
          <Bot size={20} />
          <span>AI Assistant</span>
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
        </div>
        
        <div className="ai-controls">
          <div className="execution-mode">
            <select
              value={executionMode}
              onChange={(e) => setExecutionMode(e.target.value as any)}
              className="mode-select"
            >
              <option value="review">Review Mode</option>
              <option value="auto">Auto Mode</option>
              <option value="dry-run">Dry Run</option>
            </select>
          </div>
          
          <button
            onClick={onToggleMinimize}
            className="control-btn"
            title="Minimize"
          >
            <Minimize2 size={16} />
          </button>
          
          <button className="control-btn" title="Settings">
            <Settings size={16} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="ai-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.type}`}
          >
            <div className="message-header">
              <div className="message-info">
                {message.agentType && (
                  <span className={`agent-badge ${getAgentColor(message.agentType)}`}>
                    {getAgentIcon(message.agentType)} {message.agentType}
                  </span>
                )}
                {message.confidence && (
                  <span className="confidence">
                    {Math.round(message.confidence * 100)}% confidence
                  </span>
                )}
              </div>
              <span className="timestamp">
                {message.timestamp.toLocaleTimeString()}
              </span>
            </div>
            
            <div className="message-content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={oneDark}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {/* Command execution buttons */}
            {message.commands && message.commands.length > 0 && (
              <div className="command-actions">
                <button
                  onClick={() => handleExecuteCommands(message.commands!)}
                  className="execute-btn"
                  disabled={!message.metadata?.safety_approved}
                >
                  <Play size={16} />
                  Execute Commands
                </button>
                
                {!message.metadata?.safety_approved && (
                  <span className="safety-warning">
                    <AlertTriangle size={16} />
                    Requires manual review
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="message-content">
              <div className="loading-indicator">
                <Loader2 size={20} className="animate-spin" />
                <span>AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="ai-input-form">
        <div className="input-container">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything about development..."
            className="ai-input"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="send-btn"
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>
      </form>

      <style jsx>{`
        .ai-assistant {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: var(--ai-bg, #111827);
          border: 1px solid var(--ai-border, #374151);
          border-radius: 8px;
          overflow: hidden;
        }

        .ai-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          background: var(--ai-header-bg, #1f2937);
          border-bottom: 1px solid var(--ai-border, #374151);
        }

        .ai-title {
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 600;
          color: var(--ai-text, #ffffff);
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--status-color);
        }

        .status-indicator.connected {
          --status-color: #10b981;
        }

        .status-indicator.disconnected {
          --status-color: #ef4444;
        }

        .ai-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .mode-select {
          background: var(--select-bg, #374151);
          color: var(--ai-text, #ffffff);
          border: 1px solid var(--ai-border, #4b5563);
          border-radius: 4px;
          padding: 4px 8px;
          font-size: 12px;
        }

        .control-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          background: transparent;
          color: var(--text-muted, #9ca3af);
          border: none;
          border-radius: 4px;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .control-btn:hover {
          background: var(--hover-bg, #374151);
          color: var(--ai-text, #ffffff);
        }

        .ai-messages {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .message {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .message.user .message-content {
          background: var(--user-msg-bg, #3b82f6);
          color: white;
          padding: 12px 16px;
          border-radius: 16px 16px 4px 16px;
          margin-left: auto;
          max-width: 80%;
        }

        .message.assistant .message-content,
        .message.system .message-content {
          background: var(--ai-msg-bg, #1f2937);
          color: var(--ai-text, #ffffff);
          padding: 12px 16px;
          border-radius: 16px 16px 16px 4px;
          max-width: 100%;
          border: 1px solid var(--ai-border, #374151);
        }

        .message-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 12px;
          color: var(--text-muted, #9ca3af);
        }

        .message-info {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .agent-badge {
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .confidence {
          font-size: 10px;
          color: var(--text-muted, #6b7280);
        }

        .timestamp {
          font-size: 10px;
        }

        .command-actions {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 8px;
          padding: 8px 12px;
          background: var(--action-bg, #374151);
          border-radius: 8px;
        }

        .execute-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          background: var(--execute-bg, #10b981);
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 12px;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .execute-btn:hover:not(:disabled) {
          background: var(--execute-hover, #059669);
        }

        .execute-btn:disabled {
          background: var(--disabled-bg, #6b7280);
          cursor: not-allowed;
        }

        .safety-warning {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 11px;
          color: var(--warning-color, #f59e0b);
        }

        .loading-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
          color: var(--text-muted, #9ca3af);
        }

        .ai-input-form {
          padding: 16px;
          border-top: 1px solid var(--ai-border, #374151);
        }

        .input-container {
          display: flex;
          gap: 8px;
        }

        .ai-input {
          flex: 1;
          padding: 12px 16px;
          background: var(--input-bg, #374151);
          color: var(--ai-text, #ffffff);
          border: 1px solid var(--ai-border, #4b5563);
          border-radius: 24px;
          outline: none;
          transition: border-color 0.2s;
        }

        .ai-input:focus {
          border-color: var(--focus-color, #3b82f6);
        }

        .ai-input::placeholder {
          color: var(--placeholder-color, #9ca3af);
        }

        .send-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 44px;
          height: 44px;
          background: var(--send-bg, #3b82f6);
          color: white;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .send-btn:hover:not(:disabled) {
          background: var(--send-hover, #2563eb);
        }

        .send-btn:disabled {
          background: var(--disabled-bg, #6b7280);
          cursor: not-allowed;
        }

        @media (max-width: 768px) {
          .ai-header {
            padding: 8px 12px;
          }
          
          .ai-messages {
            padding: 12px;
          }
          
          .ai-input-form {
            padding: 12px;
          }
          
          .message.user .message-content {
            max-width: 90%;
          }
        }
      `}</style>
    </div>
  );
};