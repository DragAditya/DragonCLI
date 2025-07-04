'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import { SearchAddon } from 'xterm-addon-search';
import { Unicode11Addon } from 'xterm-addon-unicode11';
import 'xterm/css/xterm.css';
import { io, Socket } from 'socket.io-client';
import { Maximize2, Minimize2, X, Settings, Copy, Paste } from 'lucide-react';

interface TerminalComponentProps {
  sandboxId: string;
  sessionId: string;
  onCommand?: (command: string) => void;
  theme?: 'dark' | 'light';
  className?: string;
}

interface TerminalTheme {
  background: string;
  foreground: string;
  cursor: string;
  selection: string;
  black: string;
  red: string;
  green: string;
  yellow: string;
  blue: string;
  magenta: string;
  cyan: string;
  white: string;
  brightBlack: string;
  brightRed: string;
  brightGreen: string;
  brightYellow: string;
  brightBlue: string;
  brightMagenta: string;
  brightCyan: string;
  brightWhite: string;
}

const darkTheme: TerminalTheme = {
  background: '#0f0f23',
  foreground: '#cccccc',
  cursor: '#00ff00',
  selection: '#3b82f640',
  black: '#000000',
  red: '#ff6b6b',
  green: '#51cf66',
  yellow: '#ffd43b',
  blue: '#339af0',
  magenta: '#f06595',
  cyan: '#22d3ee',
  white: '#ffffff',
  brightBlack: '#666666',
  brightRed: '#ff7979',
  brightGreen: '#6bcf7f',
  brightYellow: '#ffeaa7',
  brightBlue: '#74b9ff',
  brightMagenta: '#fd79a8',
  brightCyan: '#7fdbda',
  brightWhite: '#ffffff'
};

const lightTheme: TerminalTheme = {
  background: '#ffffff',
  foreground: '#333333',
  cursor: '#0066cc',
  selection: '#3b82f640',
  black: '#000000',
  red: '#d63031',
  green: '#00b894',
  yellow: '#fdcb6e',
  blue: '#0984e3',
  magenta: '#e84393',
  cyan: '#00cec9',
  white: '#ffffff',
  brightBlack: '#666666',
  brightRed: '#e17055',
  brightGreen: '#55a3ff',
  brightYellow: '#f39c12',
  brightBlue: '#3498db',
  brightMagenta: '#e84393',
  brightCyan: '#1dd1a1',
  brightWhite: '#ffffff'
};

export const TerminalComponent: React.FC<TerminalComponentProps> = ({
  sandboxId,
  sessionId,
  onCommand,
  theme = 'dark',
  className = ''
}) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const terminal = useRef<Terminal | null>(null);
  const socket = useRef<Socket | null>(null);
  const fitAddon = useRef<FitAddon | null>(null);
  const searchAddon = useRef<SearchAddon | null>(null);
  
  const [isConnected, setIsConnected] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentCommand, setCurrentCommand] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Initialize terminal
  useEffect(() => {
    if (!terminalRef.current) return;

    const terminalTheme = theme === 'dark' ? darkTheme : lightTheme;
    
    // Create terminal instance
    terminal.current = new Terminal({
      fontFamily: '"JetBrains Mono", "Fira Code", "SF Mono", "Monaco", "Inconsolata", "Fira Mono", "Droid Sans Mono", "Source Code Pro", monospace',
      fontSize: 14,
      fontWeight: 'normal',
      fontWeightBold: 'bold',
      lineHeight: 1.2,
      letterSpacing: 0,
      theme: terminalTheme,
      cursorBlink: true,
      cursorStyle: 'block',
      allowTransparency: false,
      scrollback: 10000,
      rows: 24,
      cols: 80,
      bellStyle: 'none',
      macOptionIsMeta: true,
      rightClickSelectsWord: true,
      rendererType: 'canvas',
      experimentalCharAtlas: 'dynamic'
    });

    // Add addons
    fitAddon.current = new FitAddon();
    searchAddon.current = new SearchAddon();
    
    terminal.current.loadAddon(fitAddon.current);
    terminal.current.loadAddon(new WebLinksAddon());
    terminal.current.loadAddon(searchAddon.current);
    terminal.current.loadAddon(new Unicode11Addon());
    
    terminal.current.unicode.activeVersion = '11';

    // Open terminal
    terminal.current.open(terminalRef.current);
    
    // Fit terminal to container
    setTimeout(() => {
      fitAddon.current?.fit();
    }, 100);

    return () => {
      terminal.current?.dispose();
    };
  }, [theme]);

  // WebSocket connection
  useEffect(() => {
    if (!terminal.current) return;

    // Connect to WebSocket
    socket.current = io(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000', {
      transports: ['websocket'],
      upgrade: true
    });

    socket.current.emit('join_terminal', { sandboxId, sessionId });

    socket.current.on('connect', () => {
      setIsConnected(true);
      terminal.current?.write('\r\n\x1b[32m● Connected to Terminal++\x1b[0m\r\n');
      terminal.current?.write('\x1b[36mWelcome to your secure development sandbox!\x1b[0m\r\n\r\n');
    });

    socket.current.on('disconnect', () => {
      setIsConnected(false);
      terminal.current?.write('\r\n\x1b[31m● Disconnected from server\x1b[0m\r\n');
    });

    socket.current.on('terminal_output', (data: string) => {
      terminal.current?.write(data);
    });

    socket.current.on('terminal_error', (error: string) => {
      terminal.current?.write(`\r\n\x1b[31mError: ${error}\x1b[0m\r\n`);
    });

    socket.current.on('command_result', (result: { exitCode: number; output: string; error?: string }) => {
      if (result.output) {
        terminal.current?.write(result.output);
      }
      if (result.error) {
        terminal.current?.write(`\x1b[31m${result.error}\x1b[0m\r\n`);
      }
    });

    return () => {
      socket.current?.disconnect();
    };
  }, [sandboxId, sessionId]);

  // Handle terminal input
  useEffect(() => {
    if (!terminal.current || !socket.current) return;

    let currentLine = '';

    const handleData = (data: string) => {
      const ord = data.charCodeAt(0);
      
      // Handle special keys
      if (ord === 13) { // Enter
        const command = currentLine.trim();
        if (command) {
          // Add to command history
          setCommandHistory(prev => [...prev.slice(-99), command]);
          setHistoryIndex(-1);
          
          // Send command to server
          socket.current?.emit('terminal_input', {
            type: 'command',
            data: command,
            sandboxId,
            sessionId
          });
          
          // Notify parent component
          onCommand?.(command);
        }
        
        currentLine = '';
        setCurrentCommand('');
        terminal.current?.write('\r\n');
        
      } else if (ord === 127) { // Backspace
        if (currentLine.length > 0) {
          currentLine = currentLine.slice(0, -1);
          setCurrentCommand(currentLine);
          terminal.current?.write('\b \b');
        }
        
      } else if (ord === 27) { // Escape sequences
        const seq = data.slice(1);
        
        if (seq === '[A') { // Up arrow - previous command
          if (commandHistory.length > 0) {
            const newIndex = historyIndex === -1 ? commandHistory.length - 1 : Math.max(0, historyIndex - 1);
            setHistoryIndex(newIndex);
            
            // Clear current line
            for (let i = 0; i < currentLine.length; i++) {
              terminal.current?.write('\b \b');
            }
            
            // Write previous command
            const prevCommand = commandHistory[newIndex];
            currentLine = prevCommand;
            setCurrentCommand(prevCommand);
            terminal.current?.write(prevCommand);
          }
          
        } else if (seq === '[B') { // Down arrow - next command
          if (historyIndex > -1) {
            const newIndex = historyIndex + 1;
            
            // Clear current line
            for (let i = 0; i < currentLine.length; i++) {
              terminal.current?.write('\b \b');
            }
            
            if (newIndex >= commandHistory.length) {
              setHistoryIndex(-1);
              currentLine = '';
              setCurrentCommand('');
            } else {
              setHistoryIndex(newIndex);
              const nextCommand = commandHistory[newIndex];
              currentLine = nextCommand;
              setCurrentCommand(nextCommand);
              terminal.current?.write(nextCommand);
            }
          }
          
        } else if (seq === '[C') { // Right arrow
          // Handle cursor movement (simplified)
          
        } else if (seq === '[D') { // Left arrow
          // Handle cursor movement (simplified)
        }
        
      } else if (ord >= 32 && ord <= 126) { // Printable characters
        currentLine += data;
        setCurrentCommand(currentLine);
        terminal.current?.write(data);
        
      } else if (ord === 3) { // Ctrl+C
        socket.current?.emit('terminal_input', {
          type: 'signal',
          data: 'SIGINT',
          sandboxId,
          sessionId
        });
        terminal.current?.write('^C\r\n');
        currentLine = '';
        setCurrentCommand('');
        
      } else if (ord === 4) { // Ctrl+D
        socket.current?.emit('terminal_input', {
          type: 'signal', 
          data: 'EOF',
          sandboxId,
          sessionId
        });
        
      } else if (ord === 12) { // Ctrl+L
        terminal.current?.clear();
        
      } else if (ord === 9) { // Tab
        // Handle tab completion
        socket.current?.emit('terminal_input', {
          type: 'tab_completion',
          data: currentLine,
          sandboxId,
          sessionId
        });
      }
    };

    terminal.current.onData(handleData);

    return () => {
      terminal.current?.dispose();
    };
  }, [commandHistory, historyIndex, onCommand, sandboxId, sessionId]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (fitAddon.current && terminal.current) {
        fitAddon.current.fit();
        
        // Notify server of new dimensions
        socket.current?.emit('terminal_resize', {
          cols: terminal.current.cols,
          rows: terminal.current.rows,
          sandboxId,
          sessionId
        });
      }
    };

    window.addEventListener('resize', handleResize);
    
    // Initial fit
    setTimeout(handleResize, 100);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [sandboxId, sessionId]);

  // Utility functions
  const handleCopy = useCallback(() => {
    if (terminal.current?.hasSelection()) {
      const selection = terminal.current.getSelection();
      navigator.clipboard.writeText(selection);
    }
  }, []);

  const handlePaste = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      socket.current?.emit('terminal_input', {
        type: 'paste',
        data: text,
        sandboxId,
        sessionId
      });
    } catch (error) {
      console.error('Failed to paste:', error);
    }
  }, [sandboxId, sessionId]);

  const handleClear = useCallback(() => {
    terminal.current?.clear();
  }, []);

  const handleFullscreen = useCallback(() => {
    setIsFullscreen(!isFullscreen);
    setTimeout(() => {
      fitAddon.current?.fit();
    }, 100);
  }, [isFullscreen]);

  const handleSearch = useCallback(() => {
    searchAddon.current?.findNext('search term');
  }, []);

  return (
    <div className={`terminal-container ${isFullscreen ? 'fullscreen' : ''} ${className}`}>
      {/* Terminal Header */}
      <div className="terminal-header">
        <div className="flex items-center gap-2">
          <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            <div className="indicator-dot" />
            <span className="indicator-text">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          <div className="sandbox-info">
            <span className="sandbox-id">Sandbox: {sandboxId.slice(-8)}</span>
          </div>
        </div>

        <div className="terminal-controls">
          <button
            onClick={handleCopy}
            className="control-btn"
            title="Copy selection"
            disabled={!terminal.current?.hasSelection()}
          >
            <Copy size={16} />
          </button>
          
          <button
            onClick={handlePaste}
            className="control-btn"
            title="Paste"
          >
            <Paste size={16} />
          </button>
          
          <button
            onClick={handleClear}
            className="control-btn"
            title="Clear terminal"
          >
            <X size={16} />
          </button>
          
          <button
            onClick={handleFullscreen}
            className="control-btn"
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
          
          <button
            className="control-btn"
            title="Terminal settings"
          >
            <Settings size={16} />
          </button>
        </div>
      </div>

      {/* Terminal Body */}
      <div 
        ref={terminalRef} 
        className="terminal-body"
        tabIndex={0}
        onFocus={() => terminal.current?.focus()}
      />

      {/* Terminal Footer */}
      <div className="terminal-footer">
        <div className="current-command">
          {currentCommand && (
            <span className="command-preview">
              Current: <code>{currentCommand}</code>
            </span>
          )}
        </div>
        
        <div className="terminal-stats">
          <span className="session-info">Session: {sessionId.slice(-8)}</span>
          <span className="terminal-size">
            {terminal.current ? `${terminal.current.cols}×${terminal.current.rows}` : ''}
          </span>
        </div>
      </div>
      
      <style jsx>{`
        .terminal-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: var(--terminal-bg, #0f0f23);
          border-radius: 8px;
          overflow: hidden;
          border: 1px solid var(--border-color, #374151);
        }
        
        .terminal-container.fullscreen {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 1000;
          border-radius: 0;
        }
        
        .terminal-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 12px;
          background: var(--header-bg, #1f2937);
          border-bottom: 1px solid var(--border-color, #374151);
          user-select: none;
        }
        
        .connection-indicator {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
        }
        
        .indicator-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--status-color);
        }
        
        .connection-indicator.connected {
          --status-color: #10b981;
          color: #10b981;
        }
        
        .connection-indicator.disconnected {
          --status-color: #ef4444;
          color: #ef4444;
        }
        
        .sandbox-info {
          font-size: 12px;
          color: var(--text-muted, #9ca3af);
        }
        
        .terminal-controls {
          display: flex;
          gap: 4px;
        }
        
        .control-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          border: none;
          background: transparent;
          color: var(--text-muted, #9ca3af);
          border-radius: 4px;
          cursor: pointer;
          transition: background-color 0.2s, color 0.2s;
        }
        
        .control-btn:hover:not(:disabled) {
          background: var(--hover-bg, #374151);
          color: var(--text-primary, #ffffff);
        }
        
        .control-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .terminal-body {
          flex: 1;
          overflow: hidden;
          padding: 8px;
        }
        
        .terminal-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 4px 12px;
          background: var(--footer-bg, #111827);
          border-top: 1px solid var(--border-color, #374151);
          font-size: 11px;
          color: var(--text-muted, #9ca3af);
          user-select: none;
        }
        
        .command-preview code {
          font-family: inherit;
          background: var(--code-bg, #374151);
          padding: 2px 4px;
          border-radius: 2px;
          font-size: 10px;
        }
        
        .terminal-stats {
          display: flex;
          gap: 12px;
        }
        
        @media (max-width: 768px) {
          .terminal-header {
            padding: 6px 8px;
          }
          
          .terminal-footer {
            padding: 4px 8px;
          }
          
          .sandbox-info,
          .terminal-stats {
            display: none;
          }
        }
      `}</style>
    </div>
  );
};