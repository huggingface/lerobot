import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  // State
  const [task, setTask] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isEncoding, setIsEncoding] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [robotsReady, setRobotsReady] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [currentFps, setCurrentFps] = useState(0);
  const [loopFps, setLoopFps] = useState(0);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Ready');
  const [uploadStatus, setUploadStatus] = useState(null);
  const [rampUpRemaining, setRampUpRemaining] = useState(0);
  const [movingToZero, setMovingToZero] = useState(false);
  const [configExpanded, setConfigExpanded] = useState(false);
  const [latestRepoId, setLatestRepoId] = useState(null);

  // Configuration
  const [config, setConfig] = useState({
    leader_type: 'openarms',  // 'openarms' or 'openarms_mini'
    leader_left: 'can0',
    leader_right: 'can1',
    follower_left: 'can2',
    follower_right: 'can3',
    left_wrist: '/dev/video0',
    right_wrist: '/dev/video1',
    base: '/dev/video4'
  });

  // Available options
  const [availableCameras, setAvailableCameras] = useState([]);
  const [availableUsbPorts, setAvailableUsbPorts] = useState([]);
  const canInterfaces = ['can0', 'can1', 'can2', 'can3'];

  const statusIntervalRef = useRef(null);
  const hasInitializedRef = useRef(false);

  const loadConfig = () => {
    try {
      const saved = localStorage.getItem('openarms_config');
      if (saved) {
        const loadedConfig = JSON.parse(saved);
        setConfig(prev => ({ ...prev, ...loadedConfig }));
      }
    } catch (e) {
      console.error('Load config error:', e);
    }
  };

  const saveConfig = (newConfig) => {
    try {
      localStorage.setItem('openarms_config', JSON.stringify(newConfig || config));
    } catch (e) {
      console.error('Save config error:', e);
    }
  };

  // Fetch status periodically
  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      const data = await response.json();

      setIsRecording(data.is_recording);
      setIsInitializing(data.is_initializing);
      setIsEncoding(data.is_encoding);
      setIsUploading(data.is_uploading);
      setRobotsReady(data.robots_ready);
      setElapsedTime(data.elapsed_time);
      setCurrentFps(data.current_fps || 0);
      setLoopFps(data.loop_fps || 0);
      setEpisodeCount(data.episode_count);
      setError(data.error);
      setStatusMessage(data.status_message || 'Ready');
      setUploadStatus(data.upload_status);
      setRampUpRemaining(data.ramp_up_remaining || 0);
      setMovingToZero(data.moving_to_zero || false);
      
      // Track the latest repo_id from the backend
      if (data.latest_repo_id) {
        setLatestRepoId(data.latest_repo_id);
      }

      if (data.config) {
        // Only merge server config if we don't have a saved config (first load)
        if (!localStorage.getItem('openarms_config')) {
          setConfig(prev => {
            const merged = { ...data.config, ...prev };
            localStorage.setItem('openarms_config', JSON.stringify(merged));
            return merged;
          });
        }
      }
    } catch (e) {
      console.error('Failed to fetch status:', e);
    }
  };

  const setupRobots = async () => {
    // Show warning to verify camera positions
    const confirmed = window.confirm(
      '⚠️ IMPORTANT: Before connecting robots, please verify:\n\n' +
      '📹 Check that cameras are correctly positioned:\n' +
      '   • LEFT wrist camera is actually on the LEFT arm\n' +
      '   • RIGHT wrist camera is actually on the RIGHT arm\n' +
      '   • BASE camera is actually the BASE/overhead camera\n\n' +
      'Incorrect camera positioning will result in invalid training data!\n\n' +
      'Click OK to continue with robot setup, or Cancel to review configuration.'
    );
    
    if (!confirmed) {
      return; // User cancelled, don't proceed
    }
    
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/robots/setup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to setup robots');
      }

      await response.json();
      saveConfig(config);
    } catch (e) {
      setError(`Robot setup failed: ${e.message}`);
    }
  };

  // Disconnect robots
  const disconnectRobots = async () => {
    try {
      await fetch(`${API_BASE}/robots/disconnect`, { method: 'POST' });
      setRobotsReady(false);
    } catch (e) {
      console.error('Failed to disconnect robots:', e);
    }
  };

  // Discover cameras
  const discoverCameras = async () => {
    try {
      const response = await fetch(`${API_BASE}/cameras/discover`);
      const data = await response.json();
      const cameras = data.cameras || [];
      setAvailableCameras(cameras);

      // Get list of valid camera IDs
      const validCameraIds = cameras.map(cam => String(cam.id));

      // Auto-fix config if current values are invalid or not set
      const updated = { ...config };
      let changed = false;

      // Auto-fix invalid camera config
      if (!config.left_wrist || !validCameraIds.includes(config.left_wrist)) {
        if (cameras.length >= 1) {
          updated.left_wrist = String(cameras[0].id);
          changed = true;
        }
      }

      if (!config.right_wrist || !validCameraIds.includes(config.right_wrist)) {
        if (cameras.length >= 2) {
          updated.right_wrist = String(cameras[1].id);
          changed = true;
        }
      }

      if (!config.base || !validCameraIds.includes(config.base)) {
        if (cameras.length >= 3) {
          updated.base = String(cameras[2].id);
          changed = true;
        }
      }

      if (changed) {
        setConfig(updated);
        saveConfig(updated);
      }

      if (cameras.length === 0) {
        setError('No cameras detected! Please connect cameras and refresh.');
      }
    } catch (e) {
      console.error('Failed to discover cameras:', e);
      setError(`Camera discovery failed: ${e.message}`);
    }
  };

  // Discover USB ports
  const discoverUsbPorts = async () => {
    try {
      const response = await fetch(`${API_BASE}/usb/discover`);
      const data = await response.json();
      const ports = data.ports || [];
      setAvailableUsbPorts(ports);

      // Auto-fix config if OpenArms Mini is selected and ports are invalid
      if (config.leader_type === 'openarms_mini') {
        const updated = { ...config };
        let changed = false;

        if (ports.length >= 1 && !ports.includes(config.leader_left)) {
          updated.leader_left = ports[0];
          changed = true;
        }

        if (ports.length >= 2 && !ports.includes(config.leader_right)) {
          updated.leader_right = ports[1];
          changed = true;
        }

        if (changed) {
          setConfig(updated);
          saveConfig(updated);
        }
      }

      if (ports.length === 0) {
        console.warn('No USB ports detected for OpenArms Mini');
      }
    } catch (e) {
      console.error('Failed to discover USB ports:', e);
    }
  };

  // Set task only (for pedal use)
  const setTaskOnly = async () => {
    if (!task.trim()) {
      setError('Please enter a task description');
      return;
    }

    setError(null);

    try {
      const response = await fetch(`${API_BASE}/recording/set-task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, ...config })
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to set task');
      }

      const result = await response.json();
      setStatusMessage(result.message || `Task set: ${task}`);
      saveConfig(config);
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        if (!isRecording && !isInitializing) {
          setStatusMessage('Ready');
        }
      }, 3000);
    } catch (e) {
      setError(e.message);
    }
  };

  // Start recording
  const startRecording = async () => {
    if (!task.trim()) {
      setError('Please enter a task description');
      return;
    }

    setError(null);

    try {
      const response = await fetch(`${API_BASE}/recording/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, ...config })
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to start recording');
      }

      await response.json();
      saveConfig(config);
    } catch (e) {
      setError(e.message);
    }
  };

  // Stop recording
  const stopRecording = async () => {
    try {
      const response = await fetch(`${API_BASE}/recording/stop`, {
        method: 'POST'
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to stop recording');
      }

      const data = await response.json();
      setError(null);
      // Update latest repo_id after recording
      if (data.dataset_name) {
        setLatestRepoId(`lerobot-data-collection/${data.dataset_name}`);
      }
    } catch (e) {
      setError(e.message);
    }
  };

  const deleteLatestEpisode = async () => {
    if (!latestRepoId) {
      setError('No episode to delete');
      return;
    }
    
    const confirmed = window.confirm(
      `WARNING: This will permanently delete the repository:\n\n${latestRepoId}\n\nThis action cannot be undone. Continue?`
    );
    
    if (!confirmed) {
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE}/recording/delete-latest`, { method: 'POST' });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to delete episode');
      }

      const data = await response.json();
      setLatestRepoId(null);
      setEpisodeCount(Math.max(0, episodeCount - 1));
      setStatusMessage(`Deleted: ${data.deleted_repo}`);
      
      setTimeout(() => {
        if (!isRecording && !isInitializing) {
          setStatusMessage('Ready');
        }
      }, 3000);
    } catch (e) {
      setError(`Delete failed: ${e.message}`);
    }
  };

  // Reset counter
  const resetCounter = async () => {
    try {
      await fetch(`${API_BASE}/counter/reset`, { method: 'POST' });
      setEpisodeCount(0);
    } catch (e) {
      console.error('Failed to reset counter:', e);
    }
  };

  // Move robot to zero position
  const moveToZero = async () => {
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/robots/move-to-zero`, { method: 'POST' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to move to zero position');
      }
      await response.json();
    } catch (e) {
      setError(`Move to zero failed: ${e.message}`);
    }
  };

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Update config and save
  const updateConfig = (key, value) => {
    const updated = { ...config, [key]: value };
    setConfig(updated);
    saveConfig(updated);
  };

  // Initialize on mount only
  useEffect(() => {
    // Prevent double-initialization in development
    if (hasInitializedRef.current) {
      return;
    }
    hasInitializedRef.current = true;

    loadConfig();
    discoverCameras();
    discoverUsbPorts();
    fetchStatus();
    statusIntervalRef.current = setInterval(fetchStatus, 1000);

    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only once on mount

  // Discover USB ports when leader type changes to Mini
  useEffect(() => {
    if (config.leader_type === 'openarms_mini') {
      discoverUsbPorts();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.leader_type]);

  return (
    <main>
      <header>
        <h1>OpenArms Recording</h1>
      </header>

      <div className="container">
        {/* Left Column: Configuration and Recording Control */}
        <div className="left-column">
          {/* Configuration Panel */}
          <section className="panel config-panel">
          <div
            className="config-header"
            onClick={() => setConfigExpanded(!configExpanded)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && setConfigExpanded(!configExpanded)}
          >
            <h2>⚙️ Configuration</h2>
            <span className="toggle-icon">{configExpanded ? '▼' : '▶'}</span>
          </div>

          {configExpanded && (
            <div className="config-content">
              {/* Robot Setup */}
              <div className="config-section">
                <h3>🤖 Robot Setup</h3>
                <div className="robot-setup">
                  {robotsReady ? (
                    <div className="robot-status ready">
                      <span>✅ Robots Ready - Recording will start instantly</span>
                      <button onClick={disconnectRobots} className="btn-disconnect">
                        Disconnect Robots
                      </button>
                    </div>
                  ) : (
                    <div className="robot-status not-ready">
                      <span>⚠️ Robots not initialized - Recording will take ~10 seconds</span>
                      <button
                        onClick={setupRobots}
                        disabled={isRecording || isInitializing}
                        className="btn-setup"
                      >
                        🚀 Setup Robots
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Leader Type Selection */}
              <div className="config-section">
                <h3>🎮 Leader Type</h3>
                <div className="config-grid">
                  <label style={{gridColumn: '1 / -1'}}>
                    Leader Arm Type
                    <select
                      value={config.leader_type}
                      onChange={(e) => updateConfig('leader_type', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      <option value="openarms">OpenArms (CAN Bus - Damiao Motors)</option>
                      <option value="openarms_mini">OpenArms Mini (USB - Feetech Motors)</option>
                    </select>
                  </label>
                </div>
              </div>

              {/* Leader Interfaces (CAN or USB based on type) */}
              <div className="config-section">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <h3>
                    {config.leader_type === 'openarms_mini' 
                      ? `Leader Ports (USB/Serial) ${availableUsbPorts.length > 0 ? `(${availableUsbPorts.length} detected)` : ''}` 
                      : 'Leader Interfaces (CAN)'}
                  </h3>
                  {config.leader_type === 'openarms_mini' && (
                    <button
                      onClick={discoverUsbPorts}
                      className="btn-refresh"
                      disabled={isRecording || robotsReady}
                    >
                      🔄 Refresh
                    </button>
                  )}
                </div>
                
                <div className="config-grid">
                  <label>
                    Leader Left
                    <select
                      value={config.leader_left}
                      onChange={(e) => updateConfig('leader_left', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      {config.leader_type === 'openarms_mini' ? (
                        availableUsbPorts.length > 0 ? (
                          availableUsbPorts.map((port) => (
                            <option key={port} value={port}>{port}</option>
                          ))
                        ) : (
                          <option value="">No USB ports detected</option>
                        )
                      ) : (
                        canInterfaces.map((iface) => (
                          <option key={iface} value={iface}>{iface}</option>
                        ))
                      )}
                    </select>
                  </label>

                  <label>
                    Leader Right
                    <select
                      value={config.leader_right}
                      onChange={(e) => updateConfig('leader_right', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      {config.leader_type === 'openarms_mini' ? (
                        availableUsbPorts.length > 0 ? (
                          availableUsbPorts.map((port) => (
                            <option key={port} value={port}>{port}</option>
                          ))
                        ) : (
                          <option value="">No USB ports detected</option>
                        )
                      ) : (
                        canInterfaces.map((iface) => (
                          <option key={iface} value={iface}>{iface}</option>
                        ))
                      )}
                    </select>
                  </label>
                </div>
              </div>

              {/* Follower CAN Interfaces */}
              <div className="config-section">
                <h3>Follower Interfaces (CAN)</h3>
                
                <div className="config-grid">
                  <label>
                    Follower Left
                    <select
                      value={config.follower_left}
                      onChange={(e) => updateConfig('follower_left', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      {canInterfaces.map((iface) => (
                        <option key={iface} value={iface}>{iface}</option>
                      ))}
                    </select>
                  </label>

                  <label>
                    Follower Right
                    <select
                      value={config.follower_right}
                      onChange={(e) => updateConfig('follower_right', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      {canInterfaces.map((iface) => (
                        <option key={iface} value={iface}>{iface}</option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>

              {/* Camera Configuration */}
              <div className="config-section">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <h3>Cameras {availableCameras.length > 0 && `(${availableCameras.length} detected)`}</h3>
                  <button
                    onClick={discoverCameras}
                    className="btn-refresh"
                    disabled={isRecording || robotsReady}
                  >
                    🔄 Refresh
                  </button>
                </div>
                <div className="config-grid">
                  <label>
                    Left Wrist
                    <select
                      value={config.left_wrist}
                      onChange={(e) => updateConfig('left_wrist', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      {availableCameras.map((cam) => (
                        <option key={cam.id} value={String(cam.id)}>
                          {cam.name || `Camera @ ${cam.id}`}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label>
                    Right Wrist
                    <select
                      value={config.right_wrist}
                      onChange={(e) => updateConfig('right_wrist', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      {availableCameras.map((cam) => (
                        <option key={cam.id} value={String(cam.id)}>
                          {cam.name || `Camera @ ${cam.id}`}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label>
                    Base Camera
                    <select
                      value={config.base}
                      onChange={(e) => updateConfig('base', e.target.value)}
                      disabled={isRecording || robotsReady}
                    >
                      {availableCameras.map((cam) => (
                        <option key={cam.id} value={String(cam.id)}>
                          {cam.name || `Camera @ ${cam.id}`}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Control Panel */}
        <section className="panel control-panel">
          <h2>🎬 Recording Control</h2>

          {/* Status Banner - Always show important statuses */}
          {isInitializing && (
            <div className="status-banner initializing">
              <div className="spinner"></div>
              <span>{statusMessage}</span>
            </div>
          )}

          {isEncoding && (
            <div className="status-banner encoding">
              <div className="spinner"></div>
              <span>📹 {statusMessage}</span>
            </div>
          )}

          {isUploading && (
            <div className="status-banner uploading">
              <div className="spinner"></div>
              <span>☁️ {statusMessage}</span>
            </div>
          )}

          {uploadStatus && !isRecording && !isEncoding && !isUploading && (
            <div className={`status-banner ${uploadStatus.startsWith('✓') ? 'success' : 'warning'}`}>
              <span>{uploadStatus}</span>
            </div>
          )}

          <div className="control-horizontal">
            {/* Task Input and Status */}
            <div className="control-left">
              <div className="input-group">
                <input
                  type="text"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="Task description (e.g., 'pick and place')"
                  disabled={isRecording || isInitializing || isEncoding || isUploading}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && robotsReady) {
                      setTaskOnly();
                    }
                  }}
                />
                <button
                  onClick={setTaskOnly}
                  disabled={isRecording || isInitializing || isEncoding || isUploading || !robotsReady}
                  className="btn-set-task"
                  title={!robotsReady ? 'Please setup robots first' : 'Store task for pedal use (Enter key)'}
                >
                  💾 Set Task
                </button>
                <button
                  onClick={startRecording}
                  disabled={isRecording || isInitializing || isEncoding || isUploading || !robotsReady}
                  className="btn-start"
                  title={!robotsReady ? 'Please setup robots first' : ''}
                >
                  {isInitializing
                    ? '⏳ Initializing...'
                    : isRecording
                    ? '⏺ Recording...'
                    : robotsReady
                    ? '⏺ Start Recording'
                    : '⏺ Setup Robots First'}
                </button>
              </div>

              {/* Ramp-up Countdown */}
              {isRecording && rampUpRemaining > 0 && (
                <div className="ramp-up-countdown">
                  <div className="countdown-box">
                    <div className="countdown-label">⚡ WARMING UP - PID RAMP-UP</div>
                    <div className="countdown-value">{rampUpRemaining.toFixed(1)}s</div>
                    <div className="countdown-subtitle">Recording will start automatically...</div>
                  </div>
                </div>
              )}

              {/* Recording Status - Only show after ramp-up */}
              {isRecording && rampUpRemaining <= 0 && (
                <div className="status recording recording-active">
                  <div className="indicator"></div>
                  <div className="time-display">
                    <span>{formatTime(elapsedTime)}</span>
                    <span className="fps-display">
                      Loop: {loopFps.toFixed(1)} Hz
                      {loopFps > 0 && loopFps < 29 && <span className="fps-warning"> ⚠️</span>}
                    </span>
                    <span className="fps-display">Recording: {currentFps.toFixed(1)} FPS</span>
                  </div>
                  <button onClick={stopRecording} className="btn-stop">
                    ⏹ Stop
                  </button>
                </div>
              )}
            </div>

            {/* Episode Counter */}
            <div className="control-right">
              <div className="counter">
                <div className="counter-label">Episodes Recorded</div>
                <div className="counter-value">{episodeCount}</div>
                <button onClick={resetCounter} className="btn-reset">
                  Reset
                </button>
              </div>
            </div>
          </div>

          {/* Delete Latest Episode Button */}
          {!isRecording && !isInitializing && latestRepoId && (
            <div className="delete-episode-section">
              <button 
                onClick={deleteLatestEpisode}
                className="btn-delete"
                title="Delete the latest recorded episode from HuggingFace Hub"
              >
                Delete Latest Episode
              </button>
              <div className="delete-info">Will delete: {latestRepoId}</div>
            </div>
          )}

          {/* Move to Zero Button */}
          {robotsReady && !isRecording && !isInitializing && (
            <div className="zero-position-section">
              <button 
                onClick={moveToZero} 
                disabled={movingToZero}
                className="btn-zero-large"
                title="Move both leader and follower robots to zero position (2s)"
              >
                {movingToZero ? '⏳ Moving to Zero Position...' : '🎯 Move to Zero Position (Leader + Follower)'}
              </button>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="error-box">
              ⚠️ {error}
            </div>
          )}
        </section>
        </div>

        {/* Right Column: Camera Feeds */}
        <div className="right-column">
          <section className="panel cameras">
          <h2>📹 Camera Views</h2>
          {robotsReady || isRecording || isInitializing ? (
            <div className="camera-layout">
              {/* Base camera - full width */}
              <div className="camera camera-base">
                <h3>Base Camera</h3>
                <img src={`${API_BASE}/camera/stream/base`} alt="Base Camera" />
              </div>

              {/* Wrist cameras - side by side */}
              <div className="camera-wrist-container">
                <div className="camera camera-wrist">
                  <h3>Left Wrist</h3>
                  <img src={`${API_BASE}/camera/stream/left_wrist`} alt="Left Wrist Camera" />
                </div>

                <div className="camera camera-wrist">
                  <h3>Right Wrist</h3>
                  <img src={`${API_BASE}/camera/stream/right_wrist`} alt="Right Wrist Camera" />
                </div>
              </div>
            </div>
          ) : (
            <div className="camera-placeholder">
              <p>📷 Camera feeds will appear when robots are set up</p>
              <p className="hint">Click "Setup Robots" above to preview camera feeds</p>
            </div>
          )}
        </section>
        </div>

      </div>
    </main>
  );
}

export default App;

