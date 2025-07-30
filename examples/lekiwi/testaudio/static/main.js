document.addEventListener('DOMContentLoaded', () => {
    // 初始化变量
    let socket;
    let audioContext;
    let analyser;
    let microphone;
    let isServerAudioActive = false;
    let isClientAudioActive = false;
    let serverAudioQueue = [];
    let isPlayingServerAudio = false;
    
    // DOM元素
    const serverStatus = document.getElementById('server-status');
    const clientStatus = document.getElementById('client-status');
    const toggleServerBtn = document.getElementById('toggle-server-audio');
    const toggleClientBtn = document.getElementById('toggle-client-audio');
    const serverVisualizer = document.getElementById('server-visualizer');
    const clientVisualizer = document.getElementById('client-visualizer');
    
    // 初始化WebSocket连接
    function initSocket() {
        socket = io();
        
        socket.on('connect', () => {
            console.log('已连接到服务器');
            serverStatus.textContent = '已连接';
            serverStatus.className = 'px-3 py-1 rounded-full bg-green-600 text-sm';
        });
        
        socket.on('disconnect', () => {
            console.log('与服务器断开连接');
            serverStatus.textContent = '未连接';
            serverStatus.className = 'px-3 py-1 rounded-full bg-red-600 text-sm';
        });
        
        socket.on('connection_response', (data) => {
            console.log('服务器响应:', data);
        });
        
        socket.on('server_audio', (data) => {
            if (!audioContext) return;
            
            // 处理服务端音频数据
            try {
                // 将二进制数据转换为Int16Array
                const audioData = new Int16Array(data);
                
                // 更新可视化
                updateVisualizer(serverVisualizer, audioData);
                
                // 如果服务端音频已激活，则播放音频
                if (isServerAudioActive) {
                    playServerAudio(audioData);
                }
            } catch (error) {
                console.error('处理服务端音频数据时出错:', error);
            }
        });
    }
    
    // 播放服务端音频
    function playServerAudio(int16Data) {
        if (!audioContext || isPlayingServerAudio) return;
        
        try {
            isPlayingServerAudio = true;
            
            // 将Int16Array转换为Float32Array
            const float32Data = new Float32Array(int16Data.length);
            for (let i = 0; i < int16Data.length; i++) {
                float32Data[i] = int16Data[i] / 32768.0;
            }
            
            // 创建音频缓冲区
            const audioBuffer = audioContext.createBuffer(1, float32Data.length, 16000);
            audioBuffer.copyToChannel(float32Data, 0);
            
            // 创建音频源并播放
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            
            source.onended = () => {
                isPlayingServerAudio = false;
            };
            
            source.start();
            
        } catch (error) {
            console.error('播放服务端音频时出错:', error);
            isPlayingServerAudio = false;
        }
    }
    
    // 初始化音频上下文
    function initAudioContext() {
        if (!audioContext) {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                console.log('音频上下文初始化成功');
            } catch (error) {
                console.error('初始化音频上下文失败:', error);
                alert('无法初始化音频系统，请检查浏览器权限设置。');
            }
        }
    }
    
    // 启动客户端麦克风
    async function startClientMicrophone() {
        try {
            initAudioContext();
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            microphone = audioContext.createMediaStreamSource(stream);
            
            // 使用AudioWorklet或ScriptProcessor处理音频
            if (audioContext.audioWorklet) {
                // 使用AudioWorklet（更现代的方法）
                await audioContext.audioWorklet.addModule('data:application/javascript,' + encodeURIComponent(`
                    class AudioProcessor extends AudioWorkletProcessor {
                        process(inputs, outputs, parameters) {
                            const input = inputs[0];
                            const output = outputs[0];
                            
                            if (input.length > 0) {
                                const inputChannel = input[0];
                                const outputChannel = output[0];
                                
                                for (let i = 0; i < inputChannel.length; i++) {
                                    outputChannel[i] = inputChannel[i];
                                }
                                
                                // 发送音频数据到主线程
                                this.port.postMessage(inputChannel);
                            }
                            
                            return true;
                        }
                    }
                    registerProcessor('audio-processor', AudioProcessor);
                `));
                
                const processor = new AudioWorkletNode(audioContext, 'audio-processor');
                microphone.connect(processor);
                processor.connect(audioContext.destination);
                
                processor.port.onmessage = (event) => {
                    if (!isClientAudioActive) return;
                    
                    const inputData = event.data;
                    
                    // 转换为16位PCM格式
                    const pcmData = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    
                    // 发送音频数据到服务器
                    socket.emit('client_audio', pcmData.buffer);
                    
                    // 更新可视化
                    updateVisualizer(clientVisualizer, pcmData);
                };
                
            } else {
                // 回退到ScriptProcessor（旧浏览器）
                const processor = audioContext.createScriptProcessor(1024, 1, 1);
                microphone.connect(processor);
                processor.connect(audioContext.destination);
                
                processor.onaudioprocess = (event) => {
                    if (!isClientAudioActive) return;
                    
                    const inputData = event.inputBuffer.getChannelData(0);
                    
                    // 转换为16位PCM格式
                    const pcmData = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    
                    // 发送音频数据到服务器
                    socket.emit('client_audio', pcmData.buffer);
                    
                    // 更新可视化
                    updateVisualizer(clientVisualizer, pcmData);
                };
            }
            
            clientStatus.textContent = '已开启';
            clientStatus.className = 'px-3 py-1 rounded-full bg-green-600 text-sm';
            toggleClientBtn.textContent = '关闭我的麦克风';
            toggleClientBtn.className = 'w-full bg-red-600 hover:bg-red-700 text-white py-3 rounded-lg font-medium transition';
            isClientAudioActive = true;
            
            console.log('客户端麦克风启动成功');
            
        } catch (error) {
            console.error('无法访问麦克风:', error);
            alert('无法访问麦克风，请确保已授予权限。错误: ' + error.message);
        }
    }
    
    // 停止客户端麦克风
    function stopClientMicrophone() {
        if (microphone) {
            microphone.disconnect();
            microphone = null;
        }
        
        clientStatus.textContent = '已关闭';
        clientStatus.className = 'px-3 py-1 rounded-full bg-gray-600 text-sm';
        toggleClientBtn.textContent = '启动我的麦克风';
        toggleClientBtn.className = 'w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-medium transition';
        isClientAudioActive = false;
        
        // 清除可视化
        clearVisualizer(clientVisualizer);
        console.log('客户端麦克风已关闭');
    }
    
    // 更新音频可视化
    function updateVisualizer(container, data) {
        // 清除现有内容
        container.innerHTML = '';
        
        // 创建新的可视化条
        const barCount = 30;
        const maxAmplitude = Math.max(...Array.from(data).map(Math.abs));
        
        for (let i = 0; i < barCount; i++) {
            const index = Math.floor(i * data.length / barCount);
            const amplitude = Math.abs(data[index]) || 0;
            const normalized = maxAmplitude > 0 ? amplitude / maxAmplitude : 0;
            
            const bar = document.createElement('div');
            bar.className = 'bg-indigo-500 rounded';
            bar.style.width = '4px';
            bar.style.height = `${Math.max(2, normalized * 60)}px`;
            
            container.appendChild(bar);
        }
    }
    
    // 清除可视化
    function clearVisualizer(container) {
        container.innerHTML = '';
        
        // 创建空的可视化条
        for (let i = 0; i < 30; i++) {
            const bar = document.createElement('div');
            bar.className = 'bg-gray-700 rounded';
            bar.style.width = '4px';
            bar.style.height = '2px';
            container.appendChild(bar);
        }
    }
    
    // 初始化可视化
    function initVisualizers() {
        clearVisualizer(serverVisualizer);
        clearVisualizer(clientVisualizer);
    }
    
    // 事件监听器
    toggleServerBtn.addEventListener('click', () => {
        isServerAudioActive = !isServerAudioActive;
        
        if (isServerAudioActive) {
            serverStatus.textContent = '已开启';
            serverStatus.className = 'px-3 py-1 rounded-full bg-green-600 text-sm';
            toggleServerBtn.textContent = '停止服务端麦克风';
            toggleServerBtn.className = 'w-full bg-red-600 hover:bg-red-700 text-white py-3 rounded-lg font-medium transition';
            console.log('服务端音频播放已开启');
        } else {
            serverStatus.textContent = '已停止';
            serverStatus.className = 'px-3 py-1 rounded-full bg-gray-600 text-sm';
            toggleServerBtn.textContent = '启动服务端麦克风';
            toggleServerBtn.className = 'w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-lg font-medium transition';
            clearVisualizer(serverVisualizer);
            console.log('服务端音频播放已关闭');
        }
    });
    
    toggleClientBtn.addEventListener('click', () => {
        if (isClientAudioActive) {
            stopClientMicrophone();
        } else {
            startClientMicrophone();
        }
    });
    
    // 初始化应用
    initSocket();
    initVisualizers();
    
    // 在关闭窗口时清理资源
    window.addEventListener('beforeunload', () => {
        if (socket) socket.disconnect();
        if (microphone) stopClientMicrophone();
    });
});