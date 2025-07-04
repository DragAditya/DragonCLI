# Frontend Implementation Guide

## 🎨 UI/UX Design Principles

### Design System
- **Color Scheme**: Dark mode primary with light mode support
- **Typography**: JetBrains Mono for terminal, Inter for UI
- **Spacing**: 8px grid system
- **Components**: Shadcn/UI + custom terminal components

### Responsive Design
```css
/* globals.css */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --terminal-bg: #0f0f23;
  --terminal-fg: #cccccc;
  --terminal-cursor: #00ff00;
  --ai-assistant: #3b82f6;
  --danger: #ef4444;
  --success: #10b981;
  --warning: #f59e0b;
}

.terminal-grid {
  display: grid;
  grid-template-columns: 300px 1fr;
  grid-template-rows: 60px 1fr;
  height: 100vh;
}

@media (max-width: 768px) {
  .terminal-grid {
    grid-template-columns: 1fr;
    grid-template-rows: 60px 200px 1fr;
  }
}
```

## 🖥️ Terminal Component

### Core Terminal Implementation

```tsx
// components/Terminal/TerminalComponent.tsx
import React, { useEffect, useRef, useState } from 'react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import { SearchAddon } from 'xterm-addon-search';
import { io, Socket } from 'socket.io-client';

interface TerminalComponentProps {
  sandboxId: string;
  sessionId: string;
  onCommand?: (command: string) => void;
  theme?: 'dark' | 'light';
}

export const TerminalComponent: React.FC<TerminalComponentProps> = ({
  sandboxId,
  sessionId,
  onCommand,
  theme = 'dark'
}) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const terminal = useRef<Terminal | null>(null);
  const socket = useRef<Socket | null>(null);
  const fitAddon = useRef<FitAddon | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!terminalRef.current) return;

    // Initialize terminal
    terminal.current = new Terminal({
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: 14,
      theme: {
        background: theme === 'dark' ? '#0f0f23' : '#ffffff',
        foreground: theme === 'dark' ? '#cccccc' : '#333333',
        cursor: '#00ff00',
        selection: '#3b82f6',
      },
      cursorBlink: true,
      allowTransparency: true,
      scrollback: 10000,
    });

    // Add addons
    fitAddon.current = new FitAddon();
    terminal.current.loadAddon(fitAddon.current);
    terminal.current.loadAddon(new WebLinksAddon());
    terminal.current.loadAddon(new SearchAddon());

    // Open terminal
    terminal.current.open(terminalRef.current);
    fitAddon.current.fit();

    // Setup WebSocket connection
    socket.current = io('/terminal', {
      query: { sandboxId, sessionId }
    });

    socket.current.on('connect', () => {
      setIsConnected(true);
      terminal.current?.write('\r\n\x1b[32mConnected to sandbox\x1b[0m\r\n');
    });

    socket.current.on('output', (data: string) => {
      terminal.current?.write(data);
    });

    socket.current.on('disconnect', () => {
      setIsConnected(false);
      terminal.current?.write('\r\n\x1b[31mDisconnected from sandbox\x1b[0m\r\n');
    });

    // Handle input
    terminal.current.onData((data) => {
      socket.current?.emit('input', data);
      
      // Detect commands for AI assistance
      if (data === '\r') {
        const currentLine = terminal.current?.buffer.active.getLine(
          terminal.current.buffer.active.cursorY
        )?.translateToString() || '';
        onCommand?.(currentLine.trim());
      }
    });

    // Handle resize
    const handleResize = () => {
      fitAddon.current?.fit();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      terminal.current?.dispose();
      socket.current?.disconnect();
      window.removeEventListener('resize', handleResize);
    };
  }, [sandboxId, sessionId, theme, onCommand]);

  return (
    <div className="terminal-container">
      <div className="terminal-header">
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm font-medium">Terminal - {sandboxId}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => fitAddon.current?.fit()}
            className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      <div ref={terminalRef} className="terminal-body" />
    </div>
  );
};
```

### AI Assistant Integration

```tsx
// components/Terminal/AIAssistant.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';

interface AIAssistantProps {
  sessionId: string;
  onExecuteCommand: (command: string) => void;
  context: {
    currentDirectory: string;
    lastCommand: string;
    files: string[];
  };
}

export const AIAssistant: React.FC<AIAssistantProps> = ({
  sessionId,
  onExecuteCommand,
  context
}) => {
  const [messages, setMessages] = useState<Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    commands?: string[];
  }>>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<'review' | 'auto' | 'dry-run'>('review');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      role: 'user' as const,
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          message: input,
          context,
          mode
        })
      });

      const data = await response.json();
      
      const assistantMessage = {
        role: 'assistant' as const,
        content: data.response,
        timestamp: new Date(),
        commands: data.commands
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Auto-execute if in auto mode
      if (mode === 'auto' && data.commands?.length > 0) {
        for (const command of data.commands) {
          onExecuteCommand(command);
        }
      }
    } catch (error) {
      console.error('AI chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const executeCommand = (command: string) => {
    onExecuteCommand(command);
  };

  return (
    <div className="ai-assistant">
      <div className="ai-header">
        <h3 className="font-semibold">AI Assistant</h3>
        <div className="flex gap-1">
          {['review', 'auto', 'dry-run'].map((m) => (
            <Badge
              key={m}
              variant={mode === m ? 'default' : 'secondary'}
              className="cursor-pointer text-xs"
              onClick={() => setMode(m as any)}
            >
              {m}
            </Badge>
          ))}
        </div>
      </div>

      <div className="ai-messages">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
          >
            <div className="message-content">
              <div className="message-text">{message.content}</div>
              {message.commands && (
                <div className="commands-section">
                  <p className="text-sm font-medium mb-2">Suggested commands:</p>
                  {message.commands.map((cmd, cmdIndex) => (
                    <div key={cmdIndex} className="command-item">
                      <code className="command-code">{cmd}</code>
                      <Button
                        size="sm"
                        onClick={() => executeCommand(cmd)}
                        className="execute-btn"
                      >
                        Execute
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="assistant-message">
            <div className="loading-indicator">AI is thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="ai-input">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask AI for help with commands, debugging, or explanations..."
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
        />
        <Button onClick={sendMessage} disabled={isLoading || !input.trim()}>
          Send
        </Button>
      </div>
    </div>
  );
};
```

## 🗂️ File Explorer Component

```tsx
// components/FileExplorer/FileTree.tsx
import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, File, Folder, Plus, Trash2 } from 'lucide-react';

interface FileNode {
  name: string;
  type: 'file' | 'directory';
  path: string;
  children?: FileNode[];
  size?: number;
  modified?: Date;
}

interface FileTreeProps {
  sandboxId: string;
  onFileSelect: (file: FileNode) => void;
  onFileCreate: (path: string, type: 'file' | 'directory') => void;
  onFileDelete: (path: string) => void;
}

export const FileTree: React.FC<FileTreeProps> = ({
  sandboxId,
  onFileSelect,
  onFileCreate,
  onFileDelete
}) => {
  const [tree, setTree] = useState<FileNode[]>([]);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['/']));
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  useEffect(() => {
    loadFileTree();
  }, [sandboxId]);

  const loadFileTree = async () => {
    try {
      const response = await fetch(`/api/fs/${sandboxId}/tree`);
      const data = await response.json();
      setTree(data.tree);
    } catch (error) {
      console.error('Failed to load file tree:', error);
    }
  };

  const toggleNode = (path: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedNodes(newExpanded);
  };

  const renderNode = (node: FileNode, depth: number = 0) => {
    const isExpanded = expandedNodes.has(node.path);
    const isSelected = selectedFile === node.path;
    
    return (
      <div key={node.path}>
        <div
          className={`file-tree-node ${isSelected ? 'selected' : ''}`}
          style={{ paddingLeft: `${depth * 20 + 8}px` }}
          onClick={() => {
            if (node.type === 'directory') {
              toggleNode(node.path);
            } else {
              setSelectedFile(node.path);
              onFileSelect(node);
            }
          }}
        >
          <div className="node-content">
            {node.type === 'directory' && (
              <span className="expand-icon">
                {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </span>
            )}
            <span className="node-icon">
              {node.type === 'directory' ? <Folder size={16} /> : <File size={16} />}
            </span>
            <span className="node-name">{node.name}</span>
          </div>
          
          <div className="node-actions">
            {node.type === 'directory' && (
              <>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onFileCreate(`${node.path}/newfile.txt`, 'file');
                  }}
                  className="action-btn"
                >
                  <Plus size={14} />
                </button>
              </>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onFileDelete(node.path);
              }}
              className="action-btn delete"
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>
        
        {node.type === 'directory' && isExpanded && node.children && (
          <div className="node-children">
            {node.children.map(child => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="file-tree">
      <div className="file-tree-header">
        <h3 className="font-semibold">Files</h3>
        <div className="header-actions">
          <button
            onClick={() => onFileCreate('/', 'directory')}
            className="action-btn"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>
      <div className="file-tree-content">
        {tree.map(node => renderNode(node))}
      </div>
    </div>
  );
};
```

## 🔧 Custom Hooks

### WebSocket Hook

```tsx
// hooks/useWebSocket.ts
import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface UseWebSocketOptions {
  namespace?: string;
  query?: Record<string, string>;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
}

export const useWebSocket = (
  url: string,
  options: UseWebSocketOptions = {}
) => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    const socket = io(url + (options.namespace || ''), {
      query: options.query
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      setIsConnected(true);
      setError(null);
      options.onConnect?.();
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      options.onDisconnect?.();
    });

    socket.on('error', (err) => {
      setError(err);
      options.onError?.(err);
    });

    return () => {
      socket.close();
    };
  }, [url, options.namespace]);

  const emit = (event: string, data?: any) => {
    socketRef.current?.emit(event, data);
  };

  const on = (event: string, callback: (...args: any[]) => void) => {
    socketRef.current?.on(event, callback);
    return () => {
      socketRef.current?.off(event, callback);
    };
  };

  return {
    isConnected,
    error,
    emit,
    on,
    socket: socketRef.current
  };
};
```

### Terminal Hook

```tsx
// hooks/useTerminal.ts
import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';

interface UseTerminalOptions {
  sandboxId: string;
  sessionId: string;
  onCommand?: (command: string) => void;
}

export const useTerminal = ({
  sandboxId,
  sessionId,
  onCommand
}: UseTerminalOptions) => {
  const [output, setOutput] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);
  const [currentDirectory, setCurrentDirectory] = useState('/');

  const { isConnected, emit, on } = useWebSocket('/terminal', {
    query: { sandboxId, sessionId }
  });

  useEffect(() => {
    const cleanup = on('output', (data: string) => {
      setOutput(prev => prev + data);
    });

    return cleanup;
  }, [on]);

  useEffect(() => {
    const cleanup = on('directory-change', (newDir: string) => {
      setCurrentDirectory(newDir);
    });

    return cleanup;
  }, [on]);

  useEffect(() => {
    const cleanup = on('command-start', () => {
      setIsRunning(true);
    });

    return cleanup;
  }, [on]);

  useEffect(() => {
    const cleanup = on('command-end', (exitCode: number) => {
      setIsRunning(false);
    });

    return cleanup;
  }, [on]);

  const executeCommand = useCallback((command: string) => {
    emit('command', command);
    onCommand?.(command);
  }, [emit, onCommand]);

  const sendInput = useCallback((input: string) => {
    emit('input', input);
  }, [emit]);

  const clear = useCallback(() => {
    setOutput('');
    emit('clear');
  }, [emit]);

  return {
    isConnected,
    output,
    isRunning,
    currentDirectory,
    executeCommand,
    sendInput,
    clear
  };
};
```

## 📱 Responsive Design

### Mobile Adaptations

```tsx
// components/Layout/MobileLayout.tsx
import React, { useState } from 'react';
import { Menu, X, Terminal, Files, MessageSquare, Settings } from 'lucide-react';

interface MobileLayoutProps {
  children: React.ReactNode;
  sidebar: React.ReactNode;
  terminal: React.ReactNode;
  aiAssistant: React.ReactNode;
}

export const MobileLayout: React.FC<MobileLayoutProps> = ({
  children,
  sidebar,
  terminal,
  aiAssistant
}) => {
  const [activeTab, setActiveTab] = useState<'terminal' | 'files' | 'ai'>('terminal');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const tabs = [
    { id: 'terminal', label: 'Terminal', icon: Terminal },
    { id: 'files', label: 'Files', icon: Files },
    { id: 'ai', label: 'AI', icon: MessageSquare }
  ];

  return (
    <div className="mobile-layout">
      <header className="mobile-header">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="menu-btn"
        >
          {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
        <h1 className="header-title">Terminal++</h1>
        <button className="settings-btn">
          <Settings size={24} />
        </button>
      </header>

      {sidebarOpen && (
        <div className="mobile-sidebar-overlay">
          <div className="mobile-sidebar">
            {sidebar}
          </div>
        </div>
      )}

      <main className="mobile-main">
        <div className="tab-content">
          {activeTab === 'terminal' && terminal}
          {activeTab === 'files' && sidebar}
          {activeTab === 'ai' && aiAssistant}
        </div>
      </main>

      <nav className="mobile-tabs">
        {tabs.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            >
              <Icon size={20} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
};
```

### CSS for Mobile

```css
/* styles/mobile.css */
.mobile-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.mobile-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem;
  background: var(--background);
  border-bottom: 1px solid var(--border);
  z-index: 1000;
}

.mobile-sidebar-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 999;
}

.mobile-sidebar {
  position: fixed;
  top: 0;
  left: 0;
  width: 280px;
  height: 100%;
  background: var(--background);
  transform: translateX(0);
  transition: transform 0.3s ease;
  z-index: 1000;
}

.mobile-main {
  flex: 1;
  overflow: hidden;
}

.tab-content {
  height: 100%;
  overflow: auto;
}

.mobile-tabs {
  display: flex;
  background: var(--background);
  border-top: 1px solid var(--border);
}

.tab-btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
  padding: 0.75rem;
  border: none;
  background: none;
  color: var(--muted-foreground);
  transition: color 0.2s;
}

.tab-btn.active {
  color: var(--primary);
}

.tab-btn span {
  font-size: 0.75rem;
}

@media (max-width: 768px) {
  .desktop-layout {
    display: none;
  }
  
  .mobile-layout {
    display: flex;
  }
}
```

## 🎭 Theme System

```tsx
// contexts/ThemeContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';

type Theme = 'dark' | 'light' | 'system';

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  resolvedTheme: 'dark' | 'light';
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('system');
  const [resolvedTheme, setResolvedTheme] = useState<'dark' | 'light'>('dark');

  useEffect(() => {
    const stored = localStorage.getItem('terminal-theme') as Theme;
    if (stored) {
      setTheme(stored);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('terminal-theme', theme);
    
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      setResolvedTheme(mediaQuery.matches ? 'dark' : 'light');
      
      const handler = (e: MediaQueryListEvent) => {
        setResolvedTheme(e.matches ? 'dark' : 'light');
      };
      
      mediaQuery.addEventListener('change', handler);
      return () => mediaQuery.removeEventListener('change', handler);
    } else {
      setResolvedTheme(theme as 'dark' | 'light');
    }
  }, [theme]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', resolvedTheme);
  }, [resolvedTheme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme, resolvedTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};
```

This frontend implementation provides a comprehensive foundation for the Terminal++ interface with modern React patterns, responsive design, and comprehensive terminal integration.