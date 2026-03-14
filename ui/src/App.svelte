<script>
  import { onMount, onDestroy, tick } from 'svelte';

  // ---------------------------------------------------------------------------
  // API base — works with Vite proxy (/api → localhost:8000)
  // ---------------------------------------------------------------------------
  const API = '/api';

  // ---------------------------------------------------------------------------
  // Server status
  // ---------------------------------------------------------------------------
  let mode = 'idle';       // idle | recording | teleoperation | evaluating
  let message = 'Ready';
  let serverError = null;
  let episodeCount = 0;
  let processRunning = false;
  let activeCameras = [];

  // ---------------------------------------------------------------------------
  // UI state
  // ---------------------------------------------------------------------------
  let activeTab = 'setup';
  let logsExpanded = false;
  let logs = [];
  let logContainer;

  // Discovery lists
  let robotTypes = [];
  let teleopTypes = [];
  let serialPorts = [];
  let canInterfaces = [];
  let discoveredCameras = [];

  // ---------------------------------------------------------------------------
  // Config (persisted to localStorage)
  // ---------------------------------------------------------------------------
  const CONFIG_KEY = 'lerobot_ui_config';

  let cfg = {
    // Setup — Robot
    robotType: '',
    robotPort: '',
    robotId: '',

    // Setup — Cameras
    cameras: [],   // [{name, type, index_or_path, width, height, fps}]

    // Record — Teleop
    teleopType: '',
    teleopPort: '',
    teleopId: '',

    // Record — Dataset
    repoId: '',
    singleTask: '',
    numEpisodes: 10,
    fps: 30,
    episodeTimeS: 60,
    resetTimeS: 5,
    pushToHub: true,
    private: false,
    displayData: false,

    // Eval
    policyPath: '',
    envType: 'pusht',
    evalEpisodes: 10,
    batchSize: 1,
    evalDevice: 'cpu',
  };

  function loadConfig() {
    try {
      const raw = localStorage.getItem(CONFIG_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        cfg = { ...cfg, ...saved };
      }
    } catch (_) {}
  }

  function saveConfig() {
    try {
      localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
    } catch (_) {}
  }

  // Auto-save whenever cfg changes
  $: cfg && saveConfig();

  // ---------------------------------------------------------------------------
  // Camera preview state (separate from cfg.cameras — tracks what's previewing)
  // ---------------------------------------------------------------------------
  let previewActive = false;

  // ---------------------------------------------------------------------------
  // Polling
  // ---------------------------------------------------------------------------
  let pollInterval;

  async function fetchStatus() {
    try {
      const res = await fetch(`${API}/status`);
      if (!res.ok) return;
      const data = await res.json();

      const prevMode = mode;
      mode = data.mode;
      message = data.message;
      serverError = data.error;
      episodeCount = data.episode_count;
      processRunning = data.process_running;
      activeCameras = data.active_cameras ?? [];

      // When process ends, restart preview if we had cameras
      if (prevMode !== 'idle' && mode === 'idle' && cfg.cameras.length > 0) {
        restartPreview();
      }
    } catch (_) {}
  }

  async function fetchLogs() {
    if (!logsExpanded) return;
    try {
      const res = await fetch(`${API}/logs`);
      if (!res.ok) return;
      const data = await res.json();
      logs = data.logs ?? [];
      await tick();
      if (logContainer) {
        logContainer.scrollTop = logContainer.scrollHeight;
      }
    } catch (_) {}
  }

  // ---------------------------------------------------------------------------
  // Hardware discovery
  // ---------------------------------------------------------------------------
  async function fetchRobotTypes() {
    try {
      const res = await fetch(`${API}/robots`);
      const data = await res.json();
      robotTypes = data.types ?? [];
      if (robotTypes.length && !cfg.robotType) {
        cfg.robotType = robotTypes[0];
      }
    } catch (_) {}
  }

  async function fetchTeleopTypes() {
    try {
      const res = await fetch(`${API}/teleops`);
      const data = await res.json();
      teleopTypes = data.types ?? [];
      if (teleopTypes.length && !cfg.teleopType) {
        cfg.teleopType = teleopTypes[0];
      }
    } catch (_) {}
  }

  async function fetchSerialPorts() {
    try {
      const res = await fetch(`${API}/hardware/serial-ports`);
      const data = await res.json();
      serialPorts = data.ports ?? [];
    } catch (_) {}
  }

  async function fetchCanInterfaces() {
    try {
      const res = await fetch(`${API}/hardware/can-interfaces`);
      const data = await res.json();
      canInterfaces = data.interfaces ?? [];
    } catch (_) {}
  }

  async function fetchDiscoveredCameras() {
    try {
      const res = await fetch(`${API}/hardware/cameras`);
      const data = await res.json();
      discoveredCameras = data.cameras ?? [];
    } catch (_) {}
  }

  async function refreshPorts() {
    await Promise.all([fetchSerialPorts(), fetchCanInterfaces()]);
  }

  // ---------------------------------------------------------------------------
  // Camera helpers
  // ---------------------------------------------------------------------------
  function addCamera() {
    cfg.cameras = [
      ...cfg.cameras,
      { name: `cam${cfg.cameras.length}`, type: 'opencv', index_or_path: cfg.cameras.length, width: 640, height: 480, fps: 30 },
    ];
  }

  function removeCamera(i) {
    cfg.cameras = cfg.cameras.filter((_, idx) => idx !== i);
  }

  function autoAssignCameras() {
    cfg.cameras = discoveredCameras.map((dc, i) => ({
      name: `cam${i}`,
      type: 'opencv',
      index_or_path: dc.index_or_path,
      width: 640,
      height: 480,
      fps: 30,
    }));
  }

  async function startPreview() {
    if (mode !== 'idle') return;
    try {
      await fetch(`${API}/cameras/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cameras: cfg.cameras }),
      });
      previewActive = true;
    } catch (_) {}
  }

  async function stopPreview() {
    try {
      await fetch(`${API}/cameras/stop`, { method: 'POST' });
      previewActive = false;
    } catch (_) {}
  }

  async function restartPreview() {
    if (cfg.cameras.length === 0) return;
    await startPreview();
  }

  // ---------------------------------------------------------------------------
  // Record
  // ---------------------------------------------------------------------------
  async function startRecording() {
    if (mode !== 'idle') return;
    serverError = null;
    await stopPreview();
    try {
      const res = await fetch(`${API}/record/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          robot_type: cfg.robotType,
          robot_port: cfg.robotPort,
          robot_id: cfg.robotId,
          robot_cameras: cfg.cameras,
          teleop_type: cfg.teleopType,
          teleop_port: cfg.teleopPort,
          teleop_id: cfg.teleopId,
          repo_id: cfg.repoId,
          single_task: cfg.singleTask,
          num_episodes: cfg.numEpisodes,
          fps: cfg.fps,
          episode_time_s: cfg.episodeTimeS,
          reset_time_s: cfg.resetTimeS,
          push_to_hub: cfg.pushToHub,
          private: cfg.private,
          display_data: cfg.displayData,
        }),
      });
      if (!res.ok) {
        const d = await res.json();
        serverError = d.detail ?? 'Failed to start recording';
      }
    } catch (e) {
      serverError = String(e);
    }
  }

  // ---------------------------------------------------------------------------
  // Teleop
  // ---------------------------------------------------------------------------
  async function startTeleop() {
    if (mode !== 'idle') return;
    serverError = null;
    await stopPreview();
    try {
      const res = await fetch(`${API}/teleop/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          robot_type: cfg.robotType,
          robot_port: cfg.robotPort,
          robot_cameras: cfg.cameras,
          teleop_type: cfg.teleopType,
          teleop_port: cfg.teleopPort,
          display_data: cfg.displayData,
        }),
      });
      if (!res.ok) {
        const d = await res.json();
        serverError = d.detail ?? 'Failed to start teleoperation';
      }
    } catch (e) {
      serverError = String(e);
    }
  }

  // ---------------------------------------------------------------------------
  // Eval
  // ---------------------------------------------------------------------------
  async function startEval() {
    if (mode !== 'idle') return;
    serverError = null;
    try {
      const res = await fetch(`${API}/eval/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          policy_path: cfg.policyPath,
          env_type: cfg.envType,
          n_episodes: cfg.evalEpisodes,
          batch_size: cfg.batchSize,
          device: cfg.evalDevice,
        }),
      });
      if (!res.ok) {
        const d = await res.json();
        serverError = d.detail ?? 'Failed to start evaluation';
      }
    } catch (e) {
      serverError = String(e);
    }
  }

  // ---------------------------------------------------------------------------
  // Process control
  // ---------------------------------------------------------------------------
  async function stopProcess() {
    try {
      await fetch(`${API}/process/stop`, { method: 'POST' });
    } catch (_) {}
  }

  async function killProcess() {
    if (!confirm('Force-kill the running process? Any unsaved data may be lost.')) return;
    try {
      await fetch(`${API}/process/kill`, { method: 'POST' });
    } catch (_) {}
  }

  async function resetCounter() {
    try {
      await fetch(`${API}/counter/reset`, { method: 'POST' });
      episodeCount = 0;
    } catch (_) {}
  }

  function clearLogs() {
    logs = [];
  }

  // ---------------------------------------------------------------------------
  // Computed helpers
  // ---------------------------------------------------------------------------
  $: modeBadgeClass =
    mode === 'recording'     ? 'badge badge-recording' :
    mode === 'teleoperation' ? 'badge badge-teleoperation' :
    mode === 'evaluating'    ? 'badge badge-evaluating' :
                               'badge badge-idle';

  $: modeLabel =
    mode === 'recording'     ? 'Recording' :
    mode === 'teleoperation' ? 'Teleoperation' :
    mode === 'evaluating'    ? 'Evaluating' :
                               'Idle';

  $: isActive = mode !== 'idle';
  $: camCount = cfg.cameras.length;
  $: camGridClass =
    camCount === 2 ? 'camera-grid cams-2' :
    camCount === 3 ? 'camera-grid cams-3' :
    camCount >= 4  ? 'camera-grid cams-4' :
                     'camera-grid';

  // Determine if camera feeds should be visible (preview active OR process owns cameras)
  $: showCameraFeeds = previewActive || (isActive && cfg.cameras.length > 0);
  // When a process is active it owns the cameras — show MJPEG from process or note
  $: processOwnsCameras = isActive;

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------
  onMount(async () => {
    loadConfig();
    await Promise.all([
      fetchRobotTypes(),
      fetchTeleopTypes(),
      fetchSerialPorts(),
      fetchCanInterfaces(),
      fetchDiscoveredCameras(),
      fetchStatus(),
    ]);
    pollInterval = setInterval(() => {
      fetchStatus();
      fetchLogs();
    }, 1000);
  });

  onDestroy(() => {
    clearInterval(pollInterval);
  });
</script>

<!-- =========================================================
     Layout
     ========================================================= -->

<div class="app-shell">

  <!-- ---- Header ---- -->
  <header class="app-header">
    <div class="header-left">
      <span class="logo">🤖 LeRobot</span>
      <span class={modeBadgeClass}>
        {#if mode === 'recording'}<span class="rec-dot"></span>{/if}
        {modeLabel}
      </span>
      <span class="header-message text-muted text-sm">{message}</span>
    </div>
    <div class="header-right">
      {#if isActive}
        <button class="btn-secondary btn-sm" on:click={stopProcess}>
          ⏹ Stop
        </button>
        <button class="btn-danger btn-sm" on:click={killProcess}>
          Kill
        </button>
      {/if}
    </div>
  </header>

  <!-- ---- Main content ---- -->
  <div class="app-body">

    <!-- Left: Tabbed panel -->
    <div class="left-pane">

      <!-- Tabs -->
      <nav class="tabs">
        <button class="tab-btn {activeTab === 'setup'    ? 'active' : ''}" on:click={() => activeTab = 'setup'}>Setup</button>
        <button class="tab-btn {activeTab === 'record'   ? 'active' : ''}" on:click={() => activeTab = 'record'}>Record</button>
        <button class="tab-btn {activeTab === 'teleop'   ? 'active' : ''}" on:click={() => activeTab = 'teleop'}>Teleop</button>
        <button class="tab-btn {activeTab === 'evaluate' ? 'active' : ''}" on:click={() => activeTab = 'evaluate'}>Evaluate</button>
      </nav>

      <!-- ---- SETUP TAB ---- -->
      {#if activeTab === 'setup'}
        <div class="tab-content">

          <!-- Robot Configuration -->
          <div class="panel">
            <div class="panel-header">
              <h2>Robot Configuration</h2>
            </div>

            <div class="form-row form-row-2" style="margin-bottom: 0.75rem;">
              <label>
                Robot Type
                <select bind:value={cfg.robotType} disabled={isActive}>
                  {#each robotTypes as t}
                    <option value={t}>{t}</option>
                  {/each}
                  {#if robotTypes.length === 0}
                    <option value="">— lerobot not found —</option>
                  {/if}
                </select>
              </label>

              <label>
                Robot Port
                <div style="display: flex; gap: 0.4rem;">
                  <select bind:value={cfg.robotPort} disabled={isActive} style="flex:1;">
                    <option value="">— none —</option>
                    {#each serialPorts as p}
                      <option value={p}>{p}</option>
                    {/each}
                    {#each canInterfaces as c}
                      <option value={c}>{c} (CAN)</option>
                    {/each}
                  </select>
                  <button class="btn-ghost btn-sm btn-icon" on:click={refreshPorts} title="Refresh ports">↻</button>
                </div>
              </label>
            </div>

            <label>
              Robot ID
              <input type="text" bind:value={cfg.robotId} placeholder="e.g. my_robot (optional)" disabled={isActive} />
            </label>
          </div>

          <!-- Camera Configuration -->
          <div class="panel" style="margin-top: 0.75rem;">
            <div class="panel-header" style="justify-content: space-between;">
              <h2>Camera Configuration</h2>
              <div style="display: flex; gap: 0.5rem; align-items: center;">
                <button class="btn-ghost btn-sm" on:click={fetchDiscoveredCameras} disabled={isActive}>
                  ↻ Refresh
                </button>
                <span class="text-muted text-xs">
                  {discoveredCameras.length} detected
                </span>
              </div>
            </div>

            <!-- Discovered cameras hint -->
            {#if discoveredCameras.length > 0}
              <div class="alert alert-info" style="margin-bottom: 0.75rem; font-size: 0.8rem;">
                Detected: {discoveredCameras.map(c => c.label).join(', ')}
              </div>
            {/if}

            <!-- Camera list -->
            {#each cfg.cameras as cam, i}
              <div class="camera-entry">
                <div class="form-row" style="grid-template-columns: 1fr 1fr 1fr 1fr 1fr auto; align-items: end; margin-bottom: 0.5rem;">
                  <label>
                    Name
                    <input type="text" bind:value={cam.name} disabled={isActive} placeholder="cam0" />
                  </label>
                  <label>
                    Type
                    <select bind:value={cam.type} disabled={isActive}>
                      <option value="opencv">opencv</option>
                      <option value="realsense">realsense</option>
                    </select>
                  </label>
                  <label>
                    Index / Path
                    <input type="text" bind:value={cam.index_or_path} disabled={isActive} placeholder="0 or /dev/video0" />
                  </label>
                  <label>
                    W × H
                    <div style="display: flex; gap: 0.25rem;">
                      <input type="number" bind:value={cam.width}  min="160" max="3840" disabled={isActive} />
                      <input type="number" bind:value={cam.height} min="120" max="2160" disabled={isActive} />
                    </div>
                  </label>
                  <label>
                    FPS
                    <input type="number" bind:value={cam.fps} min="1" max="120" disabled={isActive} />
                  </label>
                  <div style="display: flex; align-items: flex-end; padding-bottom: 1px;">
                    <button class="btn-danger btn-sm btn-icon" on:click={() => removeCamera(i)} disabled={isActive} title="Remove">✕</button>
                  </div>
                </div>
              </div>
            {/each}

            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem;">
              <button class="btn-secondary btn-sm" on:click={addCamera} disabled={isActive}>
                + Add Camera
              </button>
              {#if discoveredCameras.length > 0}
                <button class="btn-secondary btn-sm" on:click={autoAssignCameras} disabled={isActive}>
                  Auto-assign from detected
                </button>
              {/if}
              <div style="flex:1;"></div>
              {#if cfg.cameras.length > 0}
                {#if previewActive && mode === 'idle'}
                  <button class="btn-secondary btn-sm" on:click={stopPreview}>
                    ⏹ Stop Preview
                  </button>
                {:else if mode === 'idle'}
                  <button class="btn-primary btn-sm" on:click={startPreview}>
                    ▶ Start Preview
                  </button>
                {/if}
              {/if}
            </div>
          </div>

        </div>

      <!-- ---- RECORD TAB ---- -->
      {:else if activeTab === 'record'}
        <div class="tab-content">

          <!-- Teleop Device -->
          <div class="panel">
            <div class="panel-header"><h2>Teleop Device</h2></div>

            <div class="form-row form-row-2" style="margin-bottom: 0.75rem;">
              <label>
                Teleop Type
                <select bind:value={cfg.teleopType} disabled={isActive}>
                  {#each teleopTypes as t}
                    <option value={t}>{t}</option>
                  {/each}
                  {#if teleopTypes.length === 0}
                    <option value="">— lerobot not found —</option>
                  {/if}
                </select>
              </label>

              <label>
                Teleop Port
                <div style="display: flex; gap: 0.4rem;">
                  <select bind:value={cfg.teleopPort} disabled={isActive} style="flex:1;">
                    <option value="">— none —</option>
                    {#each serialPorts as p}
                      <option value={p}>{p}</option>
                    {/each}
                    {#each canInterfaces as c}
                      <option value={c}>{c} (CAN)</option>
                    {/each}
                  </select>
                  <button class="btn-ghost btn-sm btn-icon" on:click={refreshPorts} title="Refresh">↻</button>
                </div>
              </label>
            </div>

            <label>
              Teleop ID
              <input type="text" bind:value={cfg.teleopId} placeholder="optional" disabled={isActive} />
            </label>
          </div>

          <!-- Dataset -->
          <div class="panel" style="margin-top: 0.75rem;">
            <div class="panel-header"><h2>Dataset</h2></div>

            <label style="margin-bottom: 0.75rem;">
              Task Description
              <textarea
                bind:value={cfg.singleTask}
                rows="2"
                placeholder="e.g. pick up the red block and place it on the blue plate"
                disabled={isActive}
                style="resize: vertical;"
              ></textarea>
            </label>

            <label style="margin-bottom: 0.75rem;">
              HuggingFace Repo ID
              <input
                type="text"
                bind:value={cfg.repoId}
                placeholder="username/dataset-name"
                disabled={isActive}
                class="font-mono"
              />
            </label>

            <div class="form-row form-row-3" style="margin-bottom: 0.75rem;">
              <label>
                Num Episodes
                <input type="number" bind:value={cfg.numEpisodes} min="1" disabled={isActive} />
              </label>
              <label>
                Episode Time (s)
                <input type="number" bind:value={cfg.episodeTimeS} min="1" disabled={isActive} />
              </label>
              <label>
                Reset Time (s)
                <input type="number" bind:value={cfg.resetTimeS} min="0" disabled={isActive} />
              </label>
            </div>

            <div class="form-row form-row-2" style="margin-bottom: 1rem;">
              <label>
                FPS
                <input type="number" bind:value={cfg.fps} min="1" max="120" disabled={isActive} />
              </label>
            </div>

            <div class="toggle-row">
              <div>
                <div class="toggle-label">Push to Hub</div>
                <div class="toggle-sublabel">Upload dataset to HuggingFace after recording</div>
              </div>
              <label class="toggle">
                <input type="checkbox" bind:checked={cfg.pushToHub} disabled={isActive} />
                <span class="toggle-track"></span>
              </label>
            </div>

            <div class="toggle-row">
              <div>
                <div class="toggle-label">Private Dataset</div>
                <div class="toggle-sublabel">Make repository private on HuggingFace</div>
              </div>
              <label class="toggle">
                <input type="checkbox" bind:checked={cfg.private} disabled={isActive} />
                <span class="toggle-track"></span>
              </label>
            </div>
          </div>

          <!-- Action -->
          <div style="margin-top: 1rem;">
            {#if mode === 'recording'}
              <div class="alert alert-error" style="margin-bottom: 0.75rem; align-items: center;">
                <span class="rec-dot"></span>
                <span>Recording in progress — {message}</span>
              </div>
              <button class="btn-danger btn-lg" on:click={stopProcess}>
                ⏹ Stop Recording
              </button>
            {:else}
              <button
                class="btn-primary btn-lg"
                on:click={startRecording}
                disabled={isActive || !cfg.robotType || !cfg.teleopType || !cfg.singleTask || !cfg.repoId}
              >
                ▶ Start Recording
              </button>
            {/if}

            {#if !cfg.singleTask && mode === 'idle'}
              <p class="text-muted text-sm mt-2">Fill in task description and repo ID to enable recording.</p>
            {/if}
          </div>

        </div>

      <!-- ---- TELEOP TAB ---- -->
      {:else if activeTab === 'teleop'}
        <div class="tab-content">

          <div class="panel">
            <div class="panel-header"><h2>Teleoperation</h2></div>

            <!-- Summary of setup config -->
            <div class="config-summary">
              <div class="summary-row">
                <span class="text-muted text-sm">Robot Type</span>
                <span class="font-mono text-sm">{cfg.robotType || '—'}</span>
              </div>
              <div class="summary-row">
                <span class="text-muted text-sm">Robot Port</span>
                <span class="font-mono text-sm">{cfg.robotPort || '—'}</span>
              </div>
              <div class="summary-row">
                <span class="text-muted text-sm">Teleop Type</span>
                <span class="font-mono text-sm">{cfg.teleopType || '—'}</span>
              </div>
              <div class="summary-row">
                <span class="text-muted text-sm">Teleop Port</span>
                <span class="font-mono text-sm">{cfg.teleopPort || '—'}</span>
              </div>
              <div class="summary-row">
                <span class="text-muted text-sm">Cameras</span>
                <span class="text-sm">{cfg.cameras.length ? cfg.cameras.map(c => c.name).join(', ') : 'none'}</span>
              </div>
            </div>

            <p class="text-muted text-sm mt-3" style="margin-bottom: 1rem;">
              Configure robot and teleop device in the <strong>Setup</strong> tab before starting.
            </p>

            {#if mode === 'teleoperation'}
              <div class="alert alert-info" style="margin-bottom: 0.75rem; align-items: center;">
                <span class="spinner"></span>
                <span>Teleoperation active — {message}</span>
              </div>
              <button class="btn-danger btn-lg" on:click={stopProcess}>
                ⏹ Stop Teleoperation
              </button>
            {:else}
              <button
                class="btn-primary btn-lg"
                on:click={startTeleop}
                disabled={isActive || !cfg.robotType || !cfg.teleopType}
              >
                ▶ Start Teleoperation
              </button>
            {/if}
          </div>

        </div>

      <!-- ---- EVALUATE TAB ---- -->
      {:else if activeTab === 'evaluate'}
        <div class="tab-content">

          <div class="panel">
            <div class="panel-header"><h2>Evaluate Policy</h2></div>

            <label style="margin-bottom: 0.75rem;">
              Policy Path
              <input
                type="text"
                bind:value={cfg.policyPath}
                placeholder="lerobot/diffusion_pusht or /path/to/local/policy"
                disabled={isActive}
                class="font-mono"
              />
            </label>

            <label style="margin-bottom: 0.75rem;">
              Environment Type
              <input
                type="text"
                bind:value={cfg.envType}
                placeholder="e.g. pusht, libero_spatial, metaworld_pick_place"
                disabled={isActive}
              />
            </label>

            <div class="form-row form-row-3" style="margin-bottom: 0.75rem;">
              <label>
                Num Episodes
                <input type="number" bind:value={cfg.evalEpisodes} min="1" disabled={isActive} />
              </label>
              <label>
                Batch Size
                <input type="number" bind:value={cfg.batchSize} min="1" disabled={isActive} />
              </label>
              <label>
                Device
                <select bind:value={cfg.evalDevice} disabled={isActive}>
                  <option value="cpu">cpu</option>
                  <option value="cuda">cuda</option>
                  <option value="mps">mps</option>
                </select>
              </label>
            </div>

            {#if mode === 'evaluating'}
              <div class="alert alert-info" style="margin-bottom: 0.75rem; align-items: center;">
                <span class="spinner"></span>
                <span>Evaluating — {message}</span>
              </div>
              <button class="btn-danger btn-lg" on:click={stopProcess}>
                ⏹ Stop Evaluation
              </button>
            {:else}
              <button
                class="btn-primary btn-lg"
                on:click={startEval}
                disabled={isActive || !cfg.policyPath || !cfg.envType}
              >
                ▶ Start Evaluation
              </button>
            {/if}
          </div>

        </div>
      {/if}

      <!-- Global error banner -->
      {#if serverError}
        <div class="alert alert-error mt-3">
          <span>⚠</span>
          <span>{serverError}</span>
        </div>
      {/if}

    </div><!-- /left-pane -->

    <!-- Right: Camera grid -->
    <div class="right-pane">
      <div class="panel" style="height: 100%; display: flex; flex-direction: column;">
        <div class="panel-header" style="justify-content: space-between;">
          <h2>Camera Preview</h2>
          {#if activeCameras.length > 0}
            <span class="badge badge-idle" style="font-size: 0.7rem;">
              {activeCameras.length} live
            </span>
          {/if}
        </div>

        {#if isActive && cfg.cameras.length > 0}
          <!-- Process owns the cameras, stream from server -->
          <div class={camGridClass} style="flex:1;">
            {#each cfg.cameras as cam}
              <div class="camera-feed">
                <img
                  src="{API}/camera/stream/{cam.name}"
                  alt={cam.name}
                  on:error|once={(e) => { e.target.style.display = 'none'; }}
                />
                <div class="camera-label">{cam.name}</div>
              </div>
            {/each}
          </div>
          <div class="alert alert-error mt-3" style="font-size: 0.8rem; align-items: center;">
            <span class="rec-dot"></span>
            <span>Cameras in use by the active robot process</span>
          </div>

        {:else if previewActive && cfg.cameras.length > 0}
          <!-- Preview mode -->
          <div class={camGridClass} style="flex:1;">
            {#each cfg.cameras as cam}
              <div class="camera-feed">
                <img
                  src="{API}/camera/stream/{cam.name}"
                  alt={cam.name}
                  on:error|once={(e) => { e.target.style.display = 'none'; }}
                />
                <div class="camera-label">{cam.name}</div>
              </div>
            {/each}
          </div>

        {:else if cfg.cameras.length > 0}
          <div class="camera-placeholder" style="flex:1;">
            <span style="font-size: 2rem;">📷</span>
            <p>{cfg.cameras.length} camera{cfg.cameras.length > 1 ? 's' : ''} configured</p>
            <p class="text-xs">Click <strong>Start Preview</strong> in Setup to view live feeds.</p>
          </div>

        {:else}
          <div class="camera-placeholder" style="flex:1;">
            <span style="font-size: 2rem;">📷</span>
            <p>No cameras configured</p>
            <p class="text-xs">Add cameras in the Setup tab.</p>
          </div>
        {/if}

      </div>
    </div><!-- /right-pane -->

  </div><!-- /app-body -->

  <!-- ---- Log panel ---- -->
  <div class="log-panel">
    <button
      class="log-toggle"
      on:click={() => { logsExpanded = !logsExpanded; if (logsExpanded) fetchLogs(); }}
    >
      <span>{logsExpanded ? '▼' : '▶'} Logs</span>
      {#if logs.length > 0}
        <span class="text-faint text-xs">{logs.length} lines</span>
      {/if}
      {#if logsExpanded}
        <span style="margin-left: auto;">
          <button class="btn-ghost btn-sm" on:click|stopPropagation={clearLogs}>Clear</button>
        </span>
      {/if}
    </button>

    {#if logsExpanded}
      <div class="log-terminal" style="height: 200px;" bind:this={logContainer}>
        {#each logs as line}
          <div class="log-line">
            <span class="log-ts">{line.ts}</span>
            <span class="log-msg">{line.msg}</span>
          </div>
        {/each}
        {#if logs.length === 0}
          <span class="log-ts">No logs yet.</span>
        {/if}
      </div>
    {/if}
  </div>

  <!-- ---- Status bar ---- -->
  <footer class="status-bar">
    <span class={modeBadgeClass}>
      {#if mode === 'recording'}<span class="rec-dot"></span>{/if}
      {modeLabel}
    </span>
    <span class="text-muted text-sm status-message">{message}</span>
    <div style="flex:1;"></div>
    <span class="text-sm">Episodes: <strong>{episodeCount}</strong></span>
    <button class="btn-ghost btn-sm" on:click={resetCounter}>Reset counter</button>
  </footer>

</div><!-- /app-shell -->

<!-- =========================================================
     Styles (scoped)
     ========================================================= -->
<style>
  /* Layout */
  .app-shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* Header */
  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.25rem;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    flex-shrink: 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .header-right {
    display: flex;
    gap: 0.5rem;
  }

  .logo {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  .header-message {
    max-width: 400px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Body */
  .app-body {
    display: grid;
    grid-template-columns: 480px 1fr;
    gap: 0.75rem;
    padding: 0.75rem;
    flex: 1;
    overflow: hidden;
    min-height: 0;
  }

  .left-pane {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    min-height: 0;
  }

  .right-pane {
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  /* Tab content */
  .tab-content {
    flex: 1;
  }

  /* Config summary in teleop tab */
  .config-summary {
    background: var(--bg);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* Camera entry in setup */
  .camera-entry {
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.5rem;
    background: var(--bg);
  }

  /* Log panel */
  .log-panel {
    flex-shrink: 0;
    border-top: 1px solid var(--border);
    background: var(--panel);
  }

  .log-toggle {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    width: 100%;
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 0.8rem;
    padding: 0.5rem 1rem;
    cursor: pointer;
    text-align: left;
  }

  .log-toggle:hover {
    background: var(--panel-hover);
    transform: none;
  }

  /* Status bar */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.4rem 1rem;
    background: var(--panel);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
    font-size: 0.8rem;
  }

  .status-message {
    max-width: 300px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Responsive: stack columns on narrow screens */
  @media (max-width: 900px) {
    .app-body {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr;
    }
  }
</style>
