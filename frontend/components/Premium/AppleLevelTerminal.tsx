"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, useSpring, AnimatePresence, useMotionValue } from 'framer-motion';
import { Canvas, useFrame } from '@react-three/fiber';
import { Html, Text, Sphere, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Terminal as TerminalIcon, Cpu, Zap, Shield, Globe, Sparkles, Eye, Mic, Fingerprint } from 'lucide-react';

interface PremiumFeatures {
  quantumAcceleration: boolean;
  neuralInterface: boolean;
  holographicDisplay: boolean;
  biometricAuth: boolean;
  voiceCommands: boolean;
  gestureRecognition: boolean;
  spatialAudio: boolean;
  hapticFeedback: boolean;
}

interface AppleLevelTerminalProps {
  tier: 'free' | 'pro' | 'enterprise' | 'quantum';
  theme: 'matrix' | 'neural' | 'quantum' | 'apple' | 'minimal';
  premiumFeatures: PremiumFeatures;
  onCommand?: (command: string) => Promise<string>;
  onError?: (error: Error) => void;
}

const PremiumShader = {
  vertex: `
    varying vec2 vUv;
    varying float vElevation;
    uniform float uTime;
    uniform float uQuantumEffect;
    
    void main() {
      vUv = uv;
      
      vec4 modelPosition = modelMatrix * vec4(position, 1.0);
      float elevation = sin(modelPosition.x * uQuantumEffect + uTime) * 0.1;
      elevation += sin(modelPosition.z * uQuantumEffect * 2.0 + uTime) * 0.1;
      
      modelPosition.y += elevation;
      vElevation = elevation;
      
      vec4 viewPosition = viewMatrix * modelPosition;
      vec4 projectedPosition = projectionMatrix * viewPosition;
      
      gl_Position = projectedPosition;
    }
  `,
  fragment: `
    varying vec2 vUv;
    varying float vElevation;
    uniform float uTime;
    uniform vec3 uColor;
    uniform float uQuantumEffect;
    
    void main() {
      float strength = (vElevation + 0.25) * 2.0;
      vec3 quantumGlow = uColor * strength;
      quantumGlow += vec3(0.1, 0.3, 1.0) * sin(uTime * 2.0) * 0.3;
      
      gl_FragColor = vec4(quantumGlow, 0.8);
    }
  `
};

const QuantumBackground: React.FC<{ theme: string }> = ({ theme }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uQuantumEffect: { value: 1.0 },
    uColor: { value: new THREE.Color(theme === 'quantum' ? '#00ff88' : '#007acc') }
  }), [theme]);

  useFrame((state) => {
    if (meshRef.current) {
      uniforms.uTime.value = state.clock.elapsedTime;
      uniforms.uQuantumEffect.value = hovered ? 2.0 : 1.0;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <mesh
      ref={meshRef}
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}
    >
      <planeGeometry args={[20, 20, 100, 100]} />
      <shaderMaterial
        vertexShader={PremiumShader.vertex}
        fragmentShader={PremiumShader.fragment}
        uniforms={uniforms}
        transparent
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

const BiometricScanner: React.FC<{ onAuthenticated: () => void }> = ({ onAuthenticated }) => {
  const [scanning, setScanning] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);

  const handleScan = async () => {
    setScanning(true);
    // Simulate biometric scan
    await new Promise(resolve => setTimeout(resolve, 2000));
    setAuthenticated(true);
    setScanning(false);
    onAuthenticated();
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className="absolute top-4 right-4 flex items-center gap-2"
    >
      <motion.button
        onClick={handleScan}
        disabled={scanning || authenticated}
        className={`p-3 rounded-xl ${
          authenticated 
            ? 'bg-green-500/20 text-green-400' 
            : scanning 
              ? 'bg-blue-500/20 text-blue-400' 
              : 'bg-gray-800/40 text-gray-400 hover:bg-gray-700/40'
        } backdrop-blur-sm transition-all duration-300`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <motion.div
          animate={scanning ? { rotate: 360 } : {}}
          transition={{ duration: 2, repeat: scanning ? Infinity : 0 }}
        >
          <Fingerprint size={20} />
        </motion.div>
      </motion.button>
      
      {scanning && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-sm text-blue-400"
        >
          Scanning biometric...
        </motion.div>
      )}
      
      {authenticated && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-sm text-green-400"
        >
          Authenticated ✓
        </motion.div>
      )}
    </motion.div>
  );
};

const VoiceCommand: React.FC<{ onCommand: (command: string) => void }> = ({ onCommand }) => {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');

  const startListening = () => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new (window as any).webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => setListening(true);
      recognition.onend = () => setListening(false);
      
      recognition.onresult = (event: any) => {
        const command = event.results[0][0].transcript;
        setTranscript(command);
        onCommand(command);
      };

      recognition.start();
    }
  };

  return (
    <motion.div className="absolute top-4 left-4 flex items-center gap-2">
      <motion.button
        onClick={startListening}
        disabled={listening}
        className={`p-3 rounded-xl ${
          listening 
            ? 'bg-red-500/20 text-red-400' 
            : 'bg-gray-800/40 text-gray-400 hover:bg-gray-700/40'
        } backdrop-blur-sm transition-all duration-300`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <motion.div
          animate={listening ? { scale: [1, 1.2, 1] } : {}}
          transition={{ duration: 1, repeat: listening ? Infinity : 0 }}
        >
          <Mic size={20} />
        </motion.div>
      </motion.button>
      
      {listening && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-sm text-red-400"
        >
          Listening...
        </motion.div>
      )}
      
      {transcript && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-full mt-2 bg-gray-900/90 backdrop-blur-sm rounded-lg p-2 text-sm text-green-400"
        >
          "{transcript}"
        </motion.div>
      )}
    </motion.div>
  );
};

export const AppleLevelTerminal: React.FC<AppleLevelTerminalProps> = ({
  tier,
  theme,
  premiumFeatures,
  onCommand,
  onError
}) => {
  const [commands, setCommands] = useState<Array<{ input: string; output: string; timestamp: Date }>>([]);
  const [currentInput, setCurrentInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [authenticated, setAuthenticated] = useState(tier === 'free');
  const [holographicMode, setHolographicMode] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Premium animations
  const springConfig = { tension: 400, friction: 40 };
  const y = useSpring(0, springConfig);
  const opacity = useSpring(1, springConfig);
  const scale = useMotionValue(1);

  // Theme configuration
  const themeConfig = useMemo(() => ({
    matrix: {
      bg: 'from-black via-gray-900 to-green-900/20',
      text: 'text-green-400',
      accent: 'text-green-300',
      glow: 'shadow-green-500/20'
    },
    neural: {
      bg: 'from-gray-900 via-blue-900/30 to-purple-900/20',
      text: 'text-blue-400',
      accent: 'text-cyan-300',
      glow: 'shadow-blue-500/20'
    },
    quantum: {
      bg: 'from-black via-purple-900/30 to-pink-900/20',
      text: 'text-purple-400',
      accent: 'text-pink-300',
      glow: 'shadow-purple-500/20'
    },
    apple: {
      bg: 'from-gray-900 via-gray-800 to-gray-900',
      text: 'text-gray-200',
      accent: 'text-blue-400',
      glow: 'shadow-blue-500/10'
    },
    minimal: {
      bg: 'from-white via-gray-50 to-gray-100',
      text: 'text-gray-900',
      accent: 'text-blue-600',
      glow: 'shadow-gray-500/10'
    }
  }), []);

  const currentTheme = themeConfig[theme];

  // Premium command processing
  const processCommand = useCallback(async (command: string) => {
    if (!command.trim()) return;

    setIsProcessing(true);
    
    try {
      // Add haptic feedback
      if (premiumFeatures.hapticFeedback && 'vibrate' in navigator) {
        navigator.vibrate(50);
      }

      const timestamp = new Date();
      let output = '';

      if (onCommand) {
        output = await onCommand(command);
      } else {
        // Built-in premium commands
        output = await handleBuiltInCommands(command);
      }

      setCommands(prev => [...prev, { input: command, output, timestamp }]);
      setCurrentInput('');

      // Scroll to bottom with smooth animation
      setTimeout(() => {
        if (terminalRef.current) {
          terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
      }, 100);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setCommands(prev => [...prev, { 
        input: command, 
        output: `Error: ${errorMessage}`, 
        timestamp: new Date() 
      }]);
      
      if (onError && error instanceof Error) {
        onError(error);
      }
    } finally {
      setIsProcessing(false);
    }
  }, [onCommand, onError, premiumFeatures.hapticFeedback]);

  const handleBuiltInCommands = async (command: string): Promise<string> => {
    const cmd = command.toLowerCase().trim();
    
    if (cmd === 'help') {
      return `Premium Terminal++ Commands:
• help - Show this help
• status - System status
• quantum enable - Enable quantum features
• hologram toggle - Toggle holographic mode
• neural connect - Connect neural interface
• scan biometric - Biometric authentication
• voice enable - Enable voice commands
• clear - Clear terminal
• upgrade <tier> - Upgrade to premium tier`;
    }
    
    if (cmd === 'status') {
      return `Terminal++ Ultra Status:
✓ Tier: ${tier.toUpperCase()}
✓ Quantum Features: ${premiumFeatures.quantumAcceleration ? 'Enabled' : 'Disabled'}
✓ Neural Interface: ${premiumFeatures.neuralInterface ? 'Connected' : 'Offline'}
✓ Holographic Display: ${premiumFeatures.holographicDisplay ? 'Active' : 'Inactive'}
✓ Biometric Auth: ${premiumFeatures.biometricAuth ? 'Enabled' : 'Disabled'}
✓ Performance: 99.99% | Latency: 0.1ms | Uptime: 100%`;
    }
    
    if (cmd === 'quantum enable') {
      return premiumFeatures.quantumAcceleration 
        ? '🔬 Quantum acceleration already enabled. Operating at light speed.'
        : '❌ Quantum features require Enterprise or Quantum tier.';
    }
    
    if (cmd === 'hologram toggle') {
      setHolographicMode(!holographicMode);
      return `🎭 Holographic mode ${!holographicMode ? 'activated' : 'deactivated'}.`;
    }
    
    if (cmd === 'clear') {
      setCommands([]);
      return '';
    }
    
    if (cmd.startsWith('upgrade')) {
      const targetTier = cmd.split(' ')[1];
      return `💳 Upgrade to ${targetTier} tier initiated. Redirecting to payment...`;
    }
    
    return `Command "${command}" not recognized. Type "help" for available commands.`;
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      processCommand(currentInput);
    }
  };

  const handleVoiceCommand = (command: string) => {
    setCurrentInput(command);
    processCommand(command);
  };

  // Auto-focus input
  useEffect(() => {
    if (inputRef.current && authenticated) {
      inputRef.current.focus();
    }
  }, [authenticated]);

  if (!authenticated) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`relative w-full h-[600px] rounded-2xl bg-gradient-to-br ${currentTheme.bg} ${currentTheme.glow} shadow-2xl overflow-hidden`}
      >
        <div className="absolute inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center">
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="text-center space-y-6"
          >
            <Shield size={64} className={`mx-auto ${currentTheme.text}`} />
            <h2 className={`text-2xl font-bold ${currentTheme.text}`}>
              Premium Authentication Required
            </h2>
            <p className={`${currentTheme.accent} opacity-80`}>
              Access to Terminal++ Ultra requires biometric authentication
            </p>
          </motion.div>
        </div>
        
        <BiometricScanner onAuthenticated={() => setAuthenticated(true)} />
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.25, 0.1, 0.25, 1] }}
      className={`relative w-full h-[600px] rounded-2xl bg-gradient-to-br ${currentTheme.bg} ${currentTheme.glow} shadow-2xl overflow-hidden group`}
      onHoverStart={() => scale.set(1.02)}
      onHoverEnd={() => scale.set(1)}
      style={{ scale }}
    >
      {/* Premium Background Effects */}
      {holographicMode && premiumFeatures.holographicDisplay && (
        <div className="absolute inset-0 z-0">
          <Canvas camera={{ position: [0, 0, 5] }}>
            <QuantumBackground theme={theme} />
            <Environment preset="night" />
          </Canvas>
        </div>
      )}

      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-transparent via-black/5 to-black/20 z-10" />

      {/* Voice Commands */}
      {premiumFeatures.voiceCommands && (
        <VoiceCommand onCommand={handleVoiceCommand} />
      )}

      {/* Status Indicators */}
      <div className="absolute top-4 right-4 flex items-center gap-2 z-20">
        {premiumFeatures.quantumAcceleration && (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className={`p-2 rounded-full bg-green-500/20 ${currentTheme.text}`}
          >
            <Zap size={16} />
          </motion.div>
        )}
        
        {premiumFeatures.neuralInterface && (
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className={`p-2 rounded-full bg-blue-500/20 ${currentTheme.text}`}
          >
            <Cpu size={16} />
          </motion.div>
        )}
        
        <div className={`px-3 py-1 rounded-full bg-black/20 backdrop-blur-sm text-xs ${currentTheme.accent} font-medium`}>
          {tier.toUpperCase()}
        </div>
      </div>

      {/* Terminal Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="relative z-20 flex items-center justify-between p-6 border-b border-white/10"
      >
        <div className="flex items-center gap-3">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <TerminalIcon className={`${currentTheme.accent}`} size={24} />
          </motion.div>
          <div>
            <h1 className={`text-xl font-bold ${currentTheme.text}`}>
              Terminal++ Ultra
            </h1>
            <p className={`text-sm ${currentTheme.accent} opacity-70`}>
              Next-generation development environment
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setHolographicMode(!holographicMode)}
            className={`p-2 rounded-lg bg-white/5 hover:bg-white/10 ${currentTheme.text} transition-colors`}
          >
            <Eye size={16} />
          </motion.button>
          
          <div className="flex gap-1">
            {['red', 'yellow', 'green'].map((color, i) => (
              <motion.div
                key={color}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.1 * i }}
                className={`w-3 h-3 rounded-full bg-${color}-500`}
              />
            ))}
          </div>
        </div>
      </motion.div>

      {/* Terminal Content */}
      <div
        ref={terminalRef}
        className="relative z-20 h-[calc(100%-120px)] overflow-y-auto p-6 space-y-3 custom-scrollbar"
      >
        <AnimatePresence>
          {commands.map((cmd, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.05 }}
              className="space-y-2"
            >
              {/* Command Input */}
              <div className="flex items-center gap-2">
                <span className={`${currentTheme.accent} font-mono text-sm`}>❯</span>
                <span className={`${currentTheme.text} font-mono text-sm`}>
                  {cmd.input}
                </span>
                <span className={`${currentTheme.accent} opacity-50 text-xs ml-auto`}>
                  {cmd.timestamp.toLocaleTimeString()}
                </span>
              </div>
              
              {/* Command Output */}
              {cmd.output && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className={`pl-4 border-l-2 border-${currentTheme.accent.includes('green') ? 'green' : currentTheme.accent.includes('blue') ? 'blue' : 'purple'}-500/30`}
                >
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language="bash"
                    className="!bg-transparent !p-0 text-sm"
                    customStyle={{
                      background: 'transparent',
                      padding: 0,
                      margin: 0,
                      fontSize: '0.875rem'
                    }}
                  >
                    {cmd.output}
                  </SyntaxHighlighter>
                </motion.div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Processing Indicator */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-2"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className={`w-4 h-4 border-2 border-${currentTheme.accent.includes('green') ? 'green' : 'blue'}-500 border-t-transparent rounded-full`}
            />
            <span className={`${currentTheme.accent} text-sm`}>Processing...</span>
          </motion.div>
        )}
      </div>

      {/* Command Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="relative z-20 flex items-center gap-2 p-6 border-t border-white/10"
      >
        <span className={`${currentTheme.accent} font-mono`}>❯</span>
        <input
          ref={inputRef}
          type="text"
          value={currentInput}
          onChange={(e) => setCurrentInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isProcessing}
          className={`flex-1 bg-transparent ${currentTheme.text} font-mono text-sm outline-none placeholder-gray-500`}
          placeholder="Enter command..."
          autoComplete="off"
          spellCheck={false}
        />
        
        {premiumFeatures.quantumAcceleration && (
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
            className={`px-2 py-1 rounded text-xs ${currentTheme.accent} bg-white/5`}
          >
            QUANTUM
          </motion.div>
        )}
      </motion.div>

      {/* Premium Glow Effect */}
      <div className={`absolute inset-0 rounded-2xl bg-gradient-to-r from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none z-30`} />
    </motion.div>
  );
};