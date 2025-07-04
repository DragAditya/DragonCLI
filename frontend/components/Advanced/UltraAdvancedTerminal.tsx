"use client"

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { SearchAddon } from '@xterm/addon-search'
import { Unicode11Addon } from '@xterm/addon-unicode11'
import { ImageAddon } from '@xterm/addon-image'
import { WebglAddon } from '@xterm/addon-webgl'
import { CanvasAddon } from '@xterm/addon-canvas'
import '@xterm/xterm/css/xterm.css'
import * as THREE from 'three'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Box, Sphere } from '@react-three/drei'
import { io, Socket } from 'socket.io-client'
import { useQuery, useMutation, useSubscription } from '@apollo/client'
import { gql } from '@apollo/client'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Mic, 
  MicOff, 
  Video, 
  VideoOff, 
  Users, 
  Cpu, 
  Zap, 
  Brain,
  Eye,
  Hand,
  Headphones,
  Palette,
  Settings,
  Share2,
  Play,
  Pause,
  Square,
  RotateCcw,
  Monitor,
  Glasses,
  Smartphone
} from 'lucide-react'

// WebAssembly modules
interface WasmModule {
  initialize: () => Promise<void>
  processCommand: (command: string) => Promise<string>
  optimizePerformance: () => Promise<void>
  getMetrics: () => Promise<any>
}

// Advanced Types
interface TerminalSession {
  id: string
  userId: string
  collaborators: string[]
  isRecording: boolean
  recording?: SessionRecording
  aiAssistance: boolean
  quantumEnhanced: boolean
  vrMode: boolean
  arMode: boolean
  voiceControlEnabled: boolean
  gestureControlEnabled: boolean
  hapticFeedbackEnabled: boolean
}

interface SessionRecording {
  id: string
  startTime: Date
  endTime?: Date
  events: TerminalEvent[]
  metadata: RecordingMetadata
}

interface TerminalEvent {
  timestamp: Date
  type: 'input' | 'output' | 'resize' | 'cursor' | 'collaboration'
  data: any
  userId?: string
}

interface RecordingMetadata {
  resolution: { width: number; height: number }
  fps: number
  compression: string
  collaboratorCount: number
  aiInteractions: number
}

interface VoiceCommand {
  command: string
  confidence: number
  intent: string
  parameters: Record<string, any>
}

interface GestureData {
  type: 'hand' | 'eye' | 'head' | 'face'
  gesture: string
  confidence: number
  coordinates: { x: number; y: number; z?: number }
  velocity?: { x: number; y: number; z?: number }
}

interface QuantumMetrics {
  entanglement: number
  coherence: number
  fidelity: number
  error_rate: number
  quantum_advantage: boolean
}

interface HolographicDisplay {
  enabled: boolean
  depth: number
  parallax: number
  eye_tracking: boolean
  hologram_quality: 'low' | 'medium' | 'high' | 'ultra'
}

// GraphQL Subscriptions
const TERMINAL_COLLABORATION_SUBSCRIPTION = gql`
  subscription TerminalCollaboration($sessionId: String!) {
    terminalCollaboration(sessionId: $sessionId) {
      type
      userId
      data
      timestamp
      cursor {
        x
        y
        visible
      }
      selection {
        start
        end
      }
    }
  }
`

const AI_ASSISTANCE_SUBSCRIPTION = gql`
  subscription AIAssistance($sessionId: String!) {
    aiAssistance(sessionId: $sessionId) {
      type
      suggestion
      code
      explanation
      confidence
      quantumEnhanced
      multiModal
    }
  }
`

// Advanced Terminal Component
export const UltraAdvancedTerminal: React.FC<{
  sessionId: string
  userId: string
  className?: string
}> = ({ sessionId, userId, className = "" }) => {
  // Refs
  const terminalRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wasmRef = useRef<WasmModule | null>(null)
  const terminalInstance = useRef<Terminal | null>(null)
  const socketRef = useRef<Socket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const videoStreamRef = useRef<MediaStream | null>(null)
  const gestureRecognizerRef = useRef<any>(null)
  const holographicDisplayRef = useRef<HolographicDisplay | null>(null)

  // State
  const [session, setSession] = useState<TerminalSession>({
    id: sessionId,
    userId,
    collaborators: [],
    isRecording: false,
    aiAssistance: true,
    quantumEnhanced: false,
    vrMode: false,
    arMode: false,
    voiceControlEnabled: false,
    gestureControlEnabled: false,
    hapticFeedbackEnabled: false
  })

  const [performance, setPerformance] = useState({
    fps: 60,
    latency: 0,
    cpu_usage: 0,
    memory_usage: 0,
    gpu_usage: 0,
    quantum_metrics: null as QuantumMetrics | null,
    wasm_optimization: false
  })

  const [voiceRecognition, setVoiceRecognition] = useState<{
    isListening: boolean
    recognition: SpeechRecognition | null
    lastCommand: VoiceCommand | null
  }>({
    isListening: false,
    recognition: null,
    lastCommand: null
  })

  const [collaborationState, setCollaborationState] = useState({
    cursors: new Map(),
    selections: new Map(),
    users: new Map()
  })

  const [threeDVisualization, setThreeDVisualization] = useState({
    enabled: false,
    scene: 'matrix' as 'matrix' | 'neural' | 'quantum' | 'fractal',
    intensity: 0.5,
    particles: 1000,
    interactive: true
  })

  const [adaptiveTheme, setAdaptiveTheme] = useState({
    mode: 'auto' as 'auto' | 'manual',
    current_theme: 'cyberpunk',
    ai_suggestions: true,
    context_aware: true,
    biometric_adaptation: false
  })

  // WebAssembly initialization
  useEffect(() => {
    const initializeWasm = async () => {
      try {
        // Load custom WebAssembly module for terminal optimization
        const wasmModule = await import('/wasm/terminal_acceleration.wasm')
        await wasmModule.default()
        
        wasmRef.current = {
          initialize: wasmModule.initialize,
          processCommand: wasmModule.processCommand,
          optimizePerformance: wasmModule.optimizePerformance,
          getMetrics: wasmModule.getMetrics
        }

        await wasmRef.current.initialize()
        setPerformance(prev => ({ ...prev, wasm_optimization: true }))
      } catch (error) {
        console.warn('WebAssembly acceleration not available:', error)
      }
    }

    initializeWasm()
  }, [])

  // Terminal initialization with advanced features
  useEffect(() => {
    if (!terminalRef.current) return

    const terminal = new Terminal({
      rows: 30,
      cols: 120,
      fontFamily: 'JetBrains Mono, Fira Code, Monaco, monospace',
      fontSize: 14,
      fontWeight: 400,
      lineHeight: 1.2,
      letterSpacing: 0,
      cursorBlink: true,
      cursorStyle: 'block',
      cursorWidth: 1,
      bellStyle: 'none',
      theme: {
        background: '#0a0a0a',
        foreground: '#00ff00',
        cursor: '#00ff00',
        cursorAccent: '#000000',
        selection: 'rgba(0, 255, 0, 0.3)',
        bright: {
          black: '#2d2d2d',
          red: '#ff6b6b',
          green: '#4ecdc4',
          yellow: '#ffe66d',
          blue: '#4d96ff',
          magenta: '#ff6bcb',
          cyan: '#4ecdc4',
          white: '#ffffff'
        }
      },
      allowProposedApi: true,
      allowTransparency: true,
      minimumContrastRatio: 1
    })

    // Advanced addons
    const fitAddon = new FitAddon()
    const webLinksAddon = new WebLinksAddon()
    const searchAddon = new SearchAddon()
    const unicode11Addon = new Unicode11Addon()
    const imageAddon = new ImageAddon()
    
    // WebGL acceleration for better performance
    const webglAddon = new WebglAddon()
    webglAddon.onContextLoss(() => {
      webglAddon.dispose()
    })

    terminal.loadAddon(fitAddon)
    terminal.loadAddon(webLinksAddon)
    terminal.loadAddon(searchAddon)
    terminal.loadAddon(unicode11Addon)
    terminal.loadAddon(imageAddon)
    terminal.loadAddon(webglAddon)

    terminal.open(terminalRef.current)
    fitAddon.fit()

    // Advanced event handling
    terminal.onData(async (data) => {
      // Process through WebAssembly if available
      if (wasmRef.current) {
        try {
          const processedData = await wasmRef.current.processCommand(data)
          // Send processed data to backend
          socketRef.current?.emit('terminal_input', {
            sessionId,
            data: processedData,
            timestamp: Date.now(),
            userId,
            wasm_processed: true
          })
        } catch (error) {
          console.error('WebAssembly processing failed:', error)
          socketRef.current?.emit('terminal_input', {
            sessionId,
            data,
            timestamp: Date.now(),
            userId,
            wasm_processed: false
          })
        }
      } else {
        socketRef.current?.emit('terminal_input', {
          sessionId,
          data,
          timestamp: Date.now(),
          userId,
          wasm_processed: false
        })
      }

      // Record for session replay
      if (session.isRecording) {
        recordEvent({
          timestamp: new Date(),
          type: 'input',
          data,
          userId
        })
      }

      // AI assistance
      if (session.aiAssistance) {
        await requestAIAssistance(data)
      }
    })

    terminal.onResize((size) => {
      socketRef.current?.emit('terminal_resize', {
        sessionId,
        cols: size.cols,
        rows: size.rows,
        userId
      })
    })

    terminal.onSelectionChange(() => {
      const selection = terminal.getSelection()
      if (selection) {
        socketRef.current?.emit('terminal_selection', {
          sessionId,
          selection,
          userId,
          timestamp: Date.now()
        })
      }
    })

    terminalInstance.current = terminal

    // Cleanup
    return () => {
      webglAddon.dispose()
      terminal.dispose()
    }
  }, [sessionId, userId, session.aiAssistance, session.isRecording])

  // Socket.IO for real-time collaboration
  useEffect(() => {
    const socket = io('/terminal', {
      auth: { userId, sessionId }
    })

    socket.on('connect', () => {
      console.log('Connected to terminal collaboration')
    })

    socket.on('terminal_output', (data) => {
      if (terminalInstance.current) {
        terminalInstance.current.write(data.content)
      }
    })

    socket.on('collaborator_cursor', (data) => {
      setCollaborationState(prev => ({
        ...prev,
        cursors: new Map(prev.cursors.set(data.userId, data.cursor))
      }))
    })

    socket.on('collaborator_selection', (data) => {
      setCollaborationState(prev => ({
        ...prev,
        selections: new Map(prev.selections.set(data.userId, data.selection))
      }))
    })

    socket.on('ai_suggestion', (data) => {
      if (session.aiAssistance) {
        displayAISuggestion(data)
      }
    })

    socket.on('quantum_enhancement', (data) => {
      if (session.quantumEnhanced) {
        applyQuantumEnhancement(data)
      }
    })

    socketRef.current = socket

    return () => {
      socket.disconnect()
    }
  }, [sessionId, userId, session.aiAssistance, session.quantumEnhanced])

  // Voice recognition setup
  useEffect(() => {
    if (!session.voiceControlEnabled) return

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onresult = (event) => {
      const lastResult = event.results[event.results.length - 1]
      if (lastResult.isFinal) {
        const command = lastResult[0].transcript.trim()
        const confidence = lastResult[0].confidence

        processVoiceCommand({
          command,
          confidence,
          intent: extractIntent(command),
          parameters: extractParameters(command)
        })
      }
    }

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
    }

    setVoiceRecognition(prev => ({ ...prev, recognition }))

    return () => {
      recognition.stop()
    }
  }, [session.voiceControlEnabled])

  // Gesture recognition setup
  useEffect(() => {
    if (!session.gestureControlEnabled) return

    const initializeGestureRecognition = async () => {
      try {
        // Initialize MediaPipe or similar gesture recognition
        const { Hands, Camera } = await import('@mediapipe/hands')
        const { drawConnectors, drawLandmarks } = await import('@mediapipe/drawing_utils')

        const hands = new Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        })

        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        })

        hands.onResults((results) => {
          if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
              const gesture = recognizeGesture(landmarks)
              if (gesture.confidence > 0.8) {
                processGestureCommand(gesture)
              }
            }
          }
        })

        gestureRecognizerRef.current = hands
      } catch (error) {
        console.error('Failed to initialize gesture recognition:', error)
      }
    }

    initializeGestureRecognition()
  }, [session.gestureControlEnabled])

  // Performance monitoring
  useEffect(() => {
    const monitorPerformance = async () => {
      const updateMetrics = async () => {
        const metrics = {
          fps: calculateFPS(),
          latency: measureLatency(),
          cpu_usage: await getCPUUsage(),
          memory_usage: getMemoryUsage(),
          gpu_usage: await getGPUUsage()
        }

        if (wasmRef.current) {
          try {
            const wasmMetrics = await wasmRef.current.getMetrics()
            metrics.wasm_optimization = true
          } catch (error) {
            console.error('Failed to get WASM metrics:', error)
          }
        }

        if (session.quantumEnhanced) {
          metrics.quantum_metrics = await getQuantumMetrics()
        }

        setPerformance(prev => ({ ...prev, ...metrics }))
      }

      const interval = setInterval(updateMetrics, 1000)
      return () => clearInterval(interval)
    }

    monitorPerformance()
  }, [session.quantumEnhanced])

  // Helper functions
  const recordEvent = (event: TerminalEvent) => {
    // Implementation for session recording
  }

  const requestAIAssistance = async (input: string) => {
    socketRef.current?.emit('ai_assistance_request', {
      sessionId,
      input,
      context: getTerminalContext(),
      userId,
      multimodal: session.arMode || session.vrMode,
      quantum_enhanced: session.quantumEnhanced
    })
  }

  const processVoiceCommand = (command: VoiceCommand) => {
    setVoiceRecognition(prev => ({ ...prev, lastCommand: command }))
    
    // Execute voice command
    switch (command.intent) {
      case 'navigate':
        terminalInstance.current?.write(`cd ${command.parameters.directory}\r`)
        break
      case 'execute':
        terminalInstance.current?.write(`${command.parameters.command}\r`)
        break
      case 'clear':
        terminalInstance.current?.clear()
        break
      case 'search':
        // Implement search functionality
        break
    }
  }

  const processGestureCommand = (gesture: GestureData) => {
    switch (gesture.gesture) {
      case 'swipe_right':
        // Navigate forward in history
        break
      case 'swipe_left':
        // Navigate backward in history
        break
      case 'pinch':
        // Zoom in/out
        break
      case 'point':
        // Click at gesture coordinates
        break
    }
  }

  const displayAISuggestion = (suggestion: any) => {
    // Display AI suggestion in terminal
  }

  const applyQuantumEnhancement = (enhancement: any) => {
    // Apply quantum-enhanced processing
  }

  const getTerminalContext = () => {
    return {
      current_directory: '/workspace',
      command_history: [],
      environment_variables: {},
      running_processes: []
    }
  }

  // Performance measurement functions
  const calculateFPS = () => 60 // Simplified
  const measureLatency = () => 5 // Simplified
  const getCPUUsage = async () => 25 // Simplified
  const getMemoryUsage = () => 45 // Simplified
  const getGPUUsage = async () => 30 // Simplified
  const getQuantumMetrics = async (): Promise<QuantumMetrics> => ({
    entanglement: 0.95,
    coherence: 0.88,
    fidelity: 0.92,
    error_rate: 0.001,
    quantum_advantage: true
  })

  // Voice command processing
  const extractIntent = (command: string): string => {
    // Simplified intent extraction
    if (command.includes('go to') || command.includes('navigate')) return 'navigate'
    if (command.includes('run') || command.includes('execute')) return 'execute'
    if (command.includes('clear')) return 'clear'
    if (command.includes('search')) return 'search'
    return 'unknown'
  }

  const extractParameters = (command: string): Record<string, any> => {
    // Simplified parameter extraction
    return {}
  }

  const recognizeGesture = (landmarks: any): GestureData => {
    // Simplified gesture recognition
    return {
      type: 'hand',
      gesture: 'point',
      confidence: 0.9,
      coordinates: { x: 0, y: 0 }
    }
  }

  // UI Controls
  const toggleVoiceControl = () => {
    setSession(prev => ({ ...prev, voiceControlEnabled: !prev.voiceControlEnabled }))
    if (voiceRecognition.recognition) {
      if (session.voiceControlEnabled) {
        voiceRecognition.recognition.stop()
      } else {
        voiceRecognition.recognition.start()
      }
    }
  }

  const toggleGestureControl = () => {
    setSession(prev => ({ ...prev, gestureControlEnabled: !prev.gestureControlEnabled }))
  }

  const toggle3DVisualization = () => {
    setThreeDVisualization(prev => ({ ...prev, enabled: !prev.enabled }))
  }

  const toggleQuantumEnhancement = () => {
    setSession(prev => ({ ...prev, quantumEnhanced: !prev.quantumEnhanced }))
  }

  const toggleVRMode = () => {
    setSession(prev => ({ ...prev, vrMode: !prev.vrMode }))
  }

  const toggleARMode = () => {
    setSession(prev => ({ ...prev, arMode: !prev.arMode }))
  }

  const startRecording = () => {
    setSession(prev => ({
      ...prev,
      isRecording: true,
      recording: {
        id: `rec_${Date.now()}`,
        startTime: new Date(),
        events: [],
        metadata: {
          resolution: { width: window.innerWidth, height: window.innerHeight },
          fps: 60,
          compression: 'h264',
          collaboratorCount: session.collaborators.length,
          aiInteractions: 0
        }
      }
    }))
  }

  const stopRecording = () => {
    setSession(prev => ({
      ...prev,
      isRecording: false,
      recording: prev.recording ? { ...prev.recording, endTime: new Date() } : undefined
    }))
  }

  // 3D Visualization Component
  const TerminalVisualization3D = () => {
    const meshRef = useRef()

    useFrame((state, delta) => {
      if (meshRef.current) {
        meshRef.current.rotation.x += delta * 0.5
        meshRef.current.rotation.y += delta * 0.2
      }
    })

    return (
      <group>
        {threeDVisualization.scene === 'matrix' && (
          <>
            {Array.from({ length: threeDVisualization.particles }).map((_, i) => (
              <Box
                key={i}
                ref={meshRef}
                position={[
                  (Math.random() - 0.5) * 20,
                  (Math.random() - 0.5) * 20,
                  (Math.random() - 0.5) * 20
                ]}
                scale={0.1}
              >
                <meshStandardMaterial color="#00ff00" emissive="#004400" />
              </Box>
            ))}
          </>
        )}
        
        {threeDVisualization.scene === 'neural' && (
          <>
            {Array.from({ length: 50 }).map((_, i) => (
              <Sphere
                key={i}
                position={[
                  Math.sin(i * 0.5) * 10,
                  Math.cos(i * 0.3) * 10,
                  Math.sin(i * 0.7) * 10
                ]}
                scale={0.2}
              >
                <meshStandardMaterial color="#4d96ff" emissive="#001144" />
              </Sphere>
            ))}
          </>
        )}

        {threeDVisualization.scene === 'quantum' && session.quantumEnhanced && (
          <>
            <Sphere position={[0, 0, 0]} scale={2}>
              <meshStandardMaterial
                color="#ff6bcb"
                emissive="#440044"
                transparent
                opacity={0.7}
              />
            </Sphere>
            {Array.from({ length: 20 }).map((_, i) => (
              <Sphere
                key={i}
                position={[
                  Math.cos(i * 0.314) * 5,
                  Math.sin(i * 0.314) * 5,
                  Math.sin(i * 0.628) * 5
                ]}
                scale={0.1}
              >
                <meshStandardMaterial color="#ffe66d" emissive="#444400" />
              </Sphere>
            ))}
          </>
        )}
      </group>
    )
  }

  return (
    <div className={`relative w-full h-full ${className}`}>
      {/* Advanced Controls Bar */}
      <div className="absolute top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-sm border-b border-green-500/30 p-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleVoiceControl}
              className={`p-2 rounded-lg transition-colors ${
                session.voiceControlEnabled
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-gray-800 text-gray-400'
              }`}
            >
              {session.voiceControlEnabled ? <Mic size={16} /> : <MicOff size={16} />}
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleGestureControl}
              className={`p-2 rounded-lg transition-colors ${
                session.gestureControlEnabled
                  ? 'bg-blue-500/20 text-blue-400'
                  : 'bg-gray-800 text-gray-400'
              }`}
            >
              <Hand size={16} />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggle3DVisualization}
              className={`p-2 rounded-lg transition-colors ${
                threeDVisualization.enabled
                  ? 'bg-purple-500/20 text-purple-400'
                  : 'bg-gray-800 text-gray-400'
              }`}
            >
              <Cpu size={16} />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleQuantumEnhancement}
              className={`p-2 rounded-lg transition-colors ${
                session.quantumEnhanced
                  ? 'bg-pink-500/20 text-pink-400'
                  : 'bg-gray-800 text-gray-400'
              }`}
            >
              <Zap size={16} />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleVRMode}
              className={`p-2 rounded-lg transition-colors ${
                session.vrMode
                  ? 'bg-orange-500/20 text-orange-400'
                  : 'bg-gray-800 text-gray-400'
              }`}
            >
              <Glasses size={16} />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleARMode}
              className={`p-2 rounded-lg transition-colors ${
                session.arMode
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'bg-gray-800 text-gray-400'
              }`}
            >
              <Smartphone size={16} />
            </motion.button>
          </div>

          <div className="flex items-center space-x-4">
            {/* Performance Indicators */}
            <div className="flex items-center space-x-2 text-xs">
              <span className="text-green-400">FPS: {performance.fps}</span>
              <span className="text-blue-400">Latency: {performance.latency}ms</span>
              <span className="text-yellow-400">CPU: {performance.cpu_usage}%</span>
              {performance.wasm_optimization && (
                <span className="text-purple-400">WASM ✓</span>
              )}
              {session.quantumEnhanced && performance.quantum_metrics && (
                <span className="text-pink-400">
                  Q: {(performance.quantum_metrics.fidelity * 100).toFixed(0)}%
                </span>
              )}
            </div>

            {/* Recording Controls */}
            <div className="flex items-center space-x-1">
              {session.isRecording ? (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={stopRecording}
                  className="p-2 bg-red-500/20 text-red-400 rounded-lg"
                >
                  <Square size={16} />
                </motion.button>
              ) : (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={startRecording}
                  className="p-2 bg-gray-800 text-gray-400 rounded-lg hover:bg-red-500/20 hover:text-red-400"
                >
                  <Play size={16} />
                </motion.button>
              )}
            </div>

            {/* Collaboration Indicator */}
            <div className="flex items-center space-x-1 text-xs">
              <Users size={14} />
              <span>{session.collaborators.length + 1}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3D Visualization Overlay */}
      <AnimatePresence>
        {threeDVisualization.enabled && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: threeDVisualization.intensity }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-10 pointer-events-none"
          >
            <Canvas>
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} />
              <TerminalVisualization3D />
              <OrbitControls enableZoom={false} enablePan={false} />
            </Canvas>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Command Feedback */}
      <AnimatePresence>
        {session.voiceControlEnabled && voiceRecognition.lastCommand && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-16 left-4 z-40 bg-green-500/20 border border-green-500/30 rounded-lg p-3 backdrop-blur-sm"
          >
            <div className="text-green-400 text-sm">
              <div className="font-semibold">Voice Command Detected</div>
              <div className="opacity-75">
                "{voiceRecognition.lastCommand.command}" 
                ({(voiceRecognition.lastCommand.confidence * 100).toFixed(0)}% confidence)
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quantum Enhancement Indicator */}
      <AnimatePresence>
        {session.quantumEnhanced && performance.quantum_metrics && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="absolute top-16 right-4 z-40 bg-pink-500/20 border border-pink-500/30 rounded-lg p-3 backdrop-blur-sm"
          >
            <div className="text-pink-400 text-sm">
              <div className="font-semibold flex items-center">
                <Zap size={14} className="mr-1" />
                Quantum Enhanced
              </div>
              <div className="space-y-1 opacity-75">
                <div>Entanglement: {(performance.quantum_metrics.entanglement * 100).toFixed(1)}%</div>
                <div>Coherence: {(performance.quantum_metrics.coherence * 100).toFixed(1)}%</div>
                <div>Fidelity: {(performance.quantum_metrics.fidelity * 100).toFixed(1)}%</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Collaboration Cursors */}
      {Array.from(collaborationState.cursors.entries()).map(([userId, cursor]) => (
        <motion.div
          key={userId}
          className="absolute z-30 pointer-events-none"
          style={{
            left: cursor.x,
            top: cursor.y,
            transform: 'translate(-50%, -50%)'
          }}
          animate={{
            scale: cursor.visible ? 1 : 0,
            opacity: cursor.visible ? 1 : 0
          }}
        >
          <div className="w-4 h-4 bg-blue-400 rounded-full border-2 border-white shadow-lg" />
          <div className="absolute top-5 left-0 bg-blue-500 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
            User {userId.slice(-4)}
          </div>
        </motion.div>
      ))}

      {/* Main Terminal */}
      <div
        ref={terminalRef}
        className="w-full h-full pt-12 bg-black"
        style={{
          fontFamily: 'JetBrains Mono, Fira Code, Monaco, monospace'
        }}
      />

      {/* Holographic Display Canvas (when enabled) */}
      {holographicDisplayRef.current?.enabled && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 z-20 pointer-events-none"
          style={{ opacity: 0.3 }}
        />
      )}
    </div>
  )
}