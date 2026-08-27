// SWIFT Real-Time Forensic Audio Defense Engine

document.addEventListener('DOMContentLoaded', () => {
    let isListening = false;
    let audioCtx = null;
    let mediaStream = null;
    let processorNode = null;
    let sourceNode = null;

    let accumulatedSamples = [];
    let liveCanvasBuffer = new Float32Array(300);

    // DOM Elements - Main Detector
    const btnMic = document.getElementById('btnMic');
    const micLabel = document.getElementById('micLabel');
    const statusPill = document.getElementById('statusPill');
    const statusText = document.getElementById('statusText');
    const waveformCanvas = document.getElementById('waveformCanvas');
    const waveStatus = document.getElementById('waveStatus');

    const resultBadge = document.getElementById('resultBadge');
    const resultIconBox = document.getElementById('resultIconBox');
    const resultText = document.getElementById('resultText');

    const riskVal = document.getElementById('riskVal');
    const gaugeSubText = document.getElementById('gaugeSubText');
    const radialCircle = document.getElementById('radialCircle');
    const pReal = document.getElementById('pReal');
    const pFake = document.getElementById('pFake');
    const fillReal = document.getElementById('fillReal');
    const fillFake = document.getElementById('fillFake');

    const latFeat = document.getElementById('latFeat');
    const latInfer = document.getElementById('latInfer');
    const latTotal = document.getElementById('latTotal');

    let decisionThreshold = 0.45;
    const threshSlider = document.getElementById('threshSlider');
    const threshVal = document.getElementById('threshVal');

    if (threshSlider && threshVal) {
        threshSlider.addEventListener('input', (e) => {
            decisionThreshold = parseFloat(e.target.value);
            threshVal.textContent = decisionThreshold.toFixed(4);
        });
    }

    // High-Tech Oscilloscope Canvas Renderer
    function drawWaveform(buffer = null) {
        if (!waveformCanvas || !waveformCanvas.parentElement) return;
        const ctx = waveformCanvas.getContext('2d');
        const width = waveformCanvas.width = waveformCanvas.parentElement.clientWidth || 600;
        const height = waveformCanvas.height = waveformCanvas.parentElement.clientHeight || 160;

        ctx.clearRect(0, 0, width, height);

        // Cyber Grid Lines
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.04)';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 30) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Center zero line
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        const points = 300;
        const centerY = height / 2;

        // Waveform Path with Glow
        ctx.lineWidth = 2;
        ctx.beginPath();

        let grad = ctx.createLinearGradient(0, 0, width, 0);
        if (isListening || isTwilioStreaming) {
            grad.addColorStop(0, '#00f0ff');
            grad.addColorStop(0.5, '#38bdf8');
            grad.addColorStop(1, '#10b981');
        } else {
            grad.addColorStop(0, '#334155');
            grad.addColorStop(1, '#475569');
        }
        ctx.strokeStyle = grad;
        ctx.shadowColor = (isListening || isTwilioStreaming) ? '#00f0ff' : 'transparent';
        ctx.shadowBlur = (isListening || isTwilioStreaming) ? 10 : 0;

        for (let i = 0; i < points; i++) {
            const x = (i / points) * width;
            let amp = 0;
            if ((isListening || isTwilioStreaming) && buffer) {
                amp = buffer[i] || 0;
            }
            const y = centerY + amp * (height * 0.42);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    // Toggle Microphone Listening
    async function toggleListening() {
        if (isListening) {
            stopListening();
        } else {
            await startListening();
        }
    }

    async function startListening() {
        try {
            const constraints = {
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    channelCount: 1
                },
                video: false
            };
            mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            audioCtx = new (window.AudioContext || window.webkitAudioContext)(); // Use native sample rate!

            sourceNode = audioCtx.createMediaStreamSource(mediaStream);
            processorNode = audioCtx.createScriptProcessor(4096, 1, 1);

            accumulatedSamples = [];
            const nativeSampleRate = audioCtx.sampleRate;
            
            // We want to send chunks of 0.5 seconds
            const chunkSize = Math.floor(nativeSampleRate * 0.5);

            processorNode.onaudioprocess = (e) => {
                if (!isListening) return;
                const inputBuffer = e.inputBuffer.getChannelData(0);

                for (let i = 0; i < inputBuffer.length; i++) {
                    accumulatedSamples.push(inputBuffer[i]);
                }

                // Update live canvas rendering buffer
                const step = Math.floor(inputBuffer.length / 300);
                for (let k = 0; k < 300; k++) {
                    liveCanvasBuffer[k] = inputBuffer[k * step] || 0;
                }
                drawWaveform(liveCanvasBuffer);

                // Send 0.5s chunks to backend
                if (accumulatedSamples.length >= chunkSize) {
                    const chunkToSend = accumulatedSamples.slice(0, chunkSize);
                    accumulatedSamples = accumulatedSamples.slice(chunkSize);
                    sendChunk(chunkToSend, nativeSampleRate);
                }
            };

            sourceNode.connect(processorNode);
            processorNode.connect(audioCtx.destination);

            isListening = true;
            if (btnMic) btnMic.classList.add('listening');
            if (micLabel) micLabel.textContent = 'MONITORING ACTIVE • CLICK TO DISARM';
            if (statusPill) statusPill.classList.add('recording');
            if (statusText) statusText.textContent = 'STREAM ACTIVE';
            if (waveStatus) waveStatus.textContent = '16,000 Hz • Live Microphone Stream';

        } catch (err) {
            console.error('Microphone error:', err);
            alert('Microphone error: ' + err.message);
        }
    }

    function stopListening() {
        isListening = false;

        if (processorNode) { processorNode.disconnect(); processorNode = null; }
        if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
        if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
        if (audioCtx) { audioCtx.close(); audioCtx = null; }

        if (btnMic) btnMic.classList.remove('listening');
        if (micLabel) micLabel.textContent = 'ARM REAL-TIME STREAM DETECTOR';
        if (statusPill) statusPill.classList.remove('recording');
        if (statusText) statusText.textContent = 'STANDBY';
        if (waveStatus) waveStatus.textContent = '16,000 Hz • 16-Bit Mono';

        if (resultBadge) resultBadge.className = 'threat-badge idle';
        if (resultText) resultText.textContent = 'SYSTEM ARMED • AWAITING AUDIO';

        updateRadialGauge(0.0);
        if (pReal) pReal.textContent = '100.0%';
        if (pFake) pFake.textContent = '0.0%';
        if (fillReal) fillReal.style.width = '100%';
        if (fillFake) fillFake.style.width = '0%';
        drawWaveform(null);
    }

    async function sendChunk(chunk, sampleRate) {
        try {
            const res = await fetch('/api/stream_chunk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chunk: chunk, threshold: decisionThreshold, sample_rate: sampleRate })
            });

            if (res.ok) {
                const data = await res.json();
                updateDetectionResult(data);
            }
        } catch (err) {
            console.error('Chunk error:', err);
        }
    }

    function updateRadialGauge(probFake) {
        const circumference = 2 * Math.PI * 80; // ~502.65
        const offset = circumference - (probFake * circumference);
        if (radialCircle) {
            radialCircle.style.strokeDashoffset = offset;
            if (probFake >= decisionThreshold) {
                radialCircle.style.stroke = '#f43f5e';
                radialCircle.style.filter = 'drop-shadow(0 0 10px #f43f5e)';
            } else {
                radialCircle.style.stroke = '#10b981';
                radialCircle.style.filter = 'drop-shadow(0 0 10px #10b981)';
            }
        }

        if (riskVal) riskVal.textContent = `${(probFake * 100).toFixed(1)}%`;
        if (gaugeSubText) {
            if (probFake >= decisionThreshold) {
                gaugeSubText.textContent = 'NEURAL SPOOF DETECTED';
                gaugeSubText.style.color = '#f43f5e';
            } else {
                gaugeSubText.textContent = 'AUTHENTIC SPEECH';
                gaugeSubText.style.color = '#10b981';
            }
        }
    }

    function updateDetectionResult(data) {
        const probFake = data.p_fake;
        const probReal = data.p_real;
        const isSpoof = data.is_spoof;

        updateRadialGauge(probFake);

        if (resultBadge) {
            if (isSpoof) {
                resultBadge.className = 'threat-badge spoof';
                if (resultText) resultText.textContent = 'CRITICAL: SYNTHETIC AI VOICE DETECTED';
                if (resultIconBox) {
                    resultIconBox.innerHTML = `
                        <svg viewBox="0 0 24 24" class="verdict-svg" fill="none" stroke="currentColor" stroke-width="2">
                            <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>
                            <line x1="12" y1="8" x2="12" y2="12"/>
                            <line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                    `;
                }
            } else {
                resultBadge.className = 'threat-badge ok';
                if (resultText) resultText.textContent = 'VERIFIED: AUTHENTIC HUMAN SPEECH';
                if (resultIconBox) {
                    resultIconBox.innerHTML = `
                        <svg viewBox="0 0 24 24" class="verdict-svg" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                            <polyline points="22 4 12 14.01 9 11.01"/>
                        </svg>
                    `;
                }
            }
        }

        if (pReal) pReal.textContent = `${(probReal * 100).toFixed(1)}%`;
        if (pFake) pFake.textContent = `${(probFake * 100).toFixed(1)}%`;
        if (fillReal) fillReal.style.width = `${(probReal * 100).toFixed(1)}%`;
        if (fillFake) fillFake.style.width = `${(probFake * 100).toFixed(1)}%`;

        if (latFeat && data.feature_extraction_ms != null) latFeat.textContent = `${data.feature_extraction_ms.toFixed(1)} ms`;
        if (latInfer && data.inference_ms != null) latInfer.textContent = `${data.inference_ms.toFixed(1)} ms`;
        if (latTotal && data.total_latency_ms != null) latTotal.textContent = `${data.total_latency_ms.toFixed(1)} ms`;
    }

    // Dataset Sample Analysis Handlers
    document.querySelectorAll('.btn-cyber-scan').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const targetBtn = e.target.closest('.btn-cyber-scan');
            const sampleName = targetBtn.getAttribute('data-sample');
            const outDiv = document.getElementById(`out_${sampleName}`);
            if (!outDiv) return;

            targetBtn.disabled = true;
            targetBtn.innerHTML = '<span>SCANNING FREQUENCIES...</span>';
            outDiv.className = 'analysis-output active';
            outDiv.innerHTML = '<span style="color: var(--cyan-primary);">Extracting 6-Channel Spectrogram &amp; Executing Model...</span>';

            try {
                const res = await fetch('/api/analyze_sample', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_name: sampleName, threshold: decisionThreshold })
                });

                if (res.ok) {
                    const data = await res.json();
                    const isSpoof = data.is_spoof;
                    const statusColor = isSpoof ? 'var(--crimson-threat)' : 'var(--emerald-safe)';
                    const statusTitle = isSpoof ? 'AI DEEPFAKE DETECTED' : 'AUTHENTIC HUMAN SPEECH';

                    outDiv.innerHTML = `
                        <div style="display:flex; justify-content:space-between; font-weight:700; color:${statusColor};">
                            <span>${statusTitle}</span>
                            <span>Fake: ${(data.p_fake * 100).toFixed(1)}%</span>
                        </div>
                        <div style="color:var(--text-secondary); font-size:10px;">
                            <span>Real: ${(data.p_real * 100).toFixed(1)}% | Latency: ${data.total_latency_ms.toFixed(1)} ms</span>
                        </div>
                    `;

                    // Synchronize main gauges and threat HUD with sample scan result
                    updateDetectionResult(data);
                }
            } catch (err) {
                outDiv.innerHTML = `<span style="color:var(--crimson-threat);">Error: ${err.message}</span>`;
            } finally {
                targetBtn.disabled = false;
                targetBtn.innerHTML = '<span>EXECUTE FORENSIC SCAN</span>';
            }
        });
    });

    if (btnMic) btnMic.addEventListener('click', toggleListening);
    drawWaveform(null);

    // ==========================================================================
    // TWILIO HOLOGRAPHIC CALL SHIELD CONTROLLER
    // ==========================================================================
    const twilioFabBtn = document.getElementById('twilioFabBtn');
    const twilioCard = document.getElementById('twilioCard');
    const twilioCloseBtn = document.getElementById('twilioCloseBtn');
    const twilioCallDot = document.getElementById('twilioCallDot');
    const twilioBannerText = document.getElementById('twilioBannerText');
    const twilioStreamSid = document.getElementById('twilioStreamSid');
    const twilioLatency = document.getElementById('twilioLatency');
    const twilioPackets = document.getElementById('twilioPackets');
    const twilioVerdictPill = document.getElementById('twilioVerdictPill');
    const twilioVerdictText = document.getElementById('twilioVerdictText');
    const twilioMeterFill = document.getElementById('twilioMeterFill');
    const twilioRealPct = document.getElementById('twilioRealPct');
    const twilioFakePct = document.getElementById('twilioFakePct');
    const twilioCallerSelect = document.getElementById('twilioCallerSelect');
    const btnStartTwilioStream = document.getElementById('btnStartTwilioStream');
    const btnStopTwilioStream = document.getElementById('btnStopTwilioStream');
    const btnCopyTwiml = document.getElementById('btnCopyTwiml');

    let isTwilioStreaming = false;
    let twilioStreamTimer = null;
    let twilioPacketIndex = 0;
    let currentTwilioPackets = [];

    // Toggle card visibility
    if (twilioFabBtn && twilioCard) {
        twilioFabBtn.addEventListener('click', () => {
            twilioCard.classList.toggle('open');
        });
    }

    if (twilioCloseBtn && twilioCard) {
        twilioCloseBtn.addEventListener('click', () => {
            twilioCard.classList.remove('open');
        });
    }

    // Copy TwiML snippet
    if (btnCopyTwiml) {
        btnCopyTwiml.addEventListener('click', () => {
            const host = window.location.host || 'localhost:8000';
            const twiml = `<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Connecting to SWIFT Realtime Deepfake Protected Conference.</Say>
    <Start>
        <Stream name="SWIFT Realtime Forensic Stream" url="wss://${host}/twilio/stream">
            <Parameter name="conference" value="SWIFT-Secure-Room"/>
        </Stream>
    </Start>
    <Dial>
        <Conference statusCallbackEvent="start end join leave" statusCallback="http://${host}/twilio/conference_events">
            SWIFT-Secure-Room
        </Conference>
    </Dial>
</Response>`;
            navigator.clipboard.writeText(twiml).then(() => {
                btnCopyTwiml.textContent = 'COPIED!';
                setTimeout(() => { btnCopyTwiml.textContent = 'COPY TWIML'; }, 2000);
            }).catch(() => {
                alert('TwiML:\n' + twiml);
            });
        });
    }

    // Start streaming simulated / live Twilio call
    if (btnStartTwilioStream) {
        btnStartTwilioStream.addEventListener('click', async () => {
            const sampleName = twilioCallerSelect ? twilioCallerSelect.value : 'hf_real_1.wav';
            btnStartTwilioStream.disabled = true;
            btnStartTwilioStream.innerHTML = '<span>CONNECTING BRIDGE...</span>';

            try {
                const simRes = await fetch('/api/twilio/simulate_call', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_name: sampleName })
                });

                if (!simRes.ok) throw new Error('Failed to initialize Twilio stream');
                const simData = await simRes.json();

                currentTwilioPackets = simData.packets || [];
                twilioPacketIndex = 0;
                isTwilioStreaming = true;

                // UI State: Active Call
                if (twilioCallDot) twilioCallDot.classList.add('active');
                if (twilioBannerText) twilioBannerText.textContent = 'ACTIVE TELEPHONY STREAMING';
                if (twilioStreamSid) twilioStreamSid.textContent = simData.stream_sid;
                btnStartTwilioStream.style.display = 'none';
                if (btnStopTwilioStream) {
                    btnStopTwilioStream.style.display = 'flex';
                    btnStopTwilioStream.disabled = false;
                }
                if (waveStatus) waveStatus.textContent = '8,000 Hz G.711 µ-Law Telephony Stream';

                // Streaming Loop: Sends 200ms G.711 µ-law packet periodically
                sendNextTwilioPacket();

            } catch (err) {
                alert('Error starting Twilio stream: ' + err.message);
                btnStartTwilioStream.disabled = false;
                btnStartTwilioStream.innerHTML = '<span>STREAM TO CONFERENCE BRIDGE</span>';
            }
        });
    }

    function stopTwilioStream() {
        isTwilioStreaming = false;
        if (twilioStreamTimer) {
            clearTimeout(twilioStreamTimer);
            twilioStreamTimer = null;
        }

        if (twilioCallDot) twilioCallDot.classList.remove('active');
        if (twilioBannerText) twilioBannerText.textContent = 'STANDBY • READY TO MONITOR';
        if (btnStartTwilioStream) {
            btnStartTwilioStream.style.display = 'flex';
            btnStartTwilioStream.disabled = false;
            btnStartTwilioStream.innerHTML = '<span>STREAM TO CONFERENCE BRIDGE</span>';
        }
        if (btnStopTwilioStream) {
            btnStopTwilioStream.style.display = 'none';
        }
        if (waveStatus) waveStatus.textContent = '16,000 Hz • 16-Bit Mono';
        drawWaveform(null);
    }

    if (btnStopTwilioStream) {
        btnStopTwilioStream.addEventListener('click', stopTwilioStream);
    }

    async function sendNextTwilioPacket() {
        if (!isTwilioStreaming || currentTwilioPackets.length === 0) return;

        if (twilioPacketIndex >= currentTwilioPackets.length) {
            // Loop back to keep streaming continuous
            twilioPacketIndex = 0;
        }

        const packet = currentTwilioPackets[twilioPacketIndex];
        twilioPacketIndex++;

        try {
            const postRes = await fetch('/api/twilio/stream_chunk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    streamSid: packet.streamSid,
                    callSid: packet.callSid,
                    media: packet.media,
                    threshold: decisionThreshold
                })
            });

            if (postRes.ok) {
                const resData = await postRes.json();
                updateTwilioTelemetry(resData);
            }
        } catch (e) {
            console.error('Twilio stream packet error:', e);
        }

        if (isTwilioStreaming) {
            twilioStreamTimer = setTimeout(sendNextTwilioPacket, 200); // 200ms frame interval
        }
    }

    function updateTwilioTelemetry(data) {
        const isSpoof = data.is_spoof;
        const pFake = data.p_fake;
        const pReal = data.p_real;

        if (twilioLatency && data.total_latency_ms != null) twilioLatency.textContent = `${data.total_latency_ms.toFixed(1)} ms`;
        if (twilioPackets && data.total_packets != null) twilioPackets.textContent = `${data.total_packets} pkts`;

        // Update In-Call Verdict Pill
        if (twilioVerdictPill) {
            if (isSpoof) {
                twilioVerdictPill.className = 'tw-verdict-pill spoof';
                if (twilioVerdictText) twilioVerdictText.textContent = 'AI VOICE CLONE DETECTED';
            } else {
                twilioVerdictPill.className = 'tw-verdict-pill ok';
                if (twilioVerdictText) twilioVerdictText.textContent = 'AUTHENTIC HUMAN SPEECH';
            }
        }

        if (twilioMeterFill) twilioMeterFill.style.width = `${(pFake * 100).toFixed(1)}%`;
        if (twilioRealPct) twilioRealPct.textContent = `${(pReal * 100).toFixed(1)}%`;
        if (twilioFakePct) twilioFakePct.textContent = `${(pFake * 100).toFixed(1)}%`;

        // Also update main dashboard gauges & verdict
        updateDetectionResult(data);

        // Simulate live audio waveform for Twilio frames
        for (let k = 0; k < 300; k++) {
            liveCanvasBuffer[k] = (Math.sin(k * 0.12 + twilioPacketIndex * 0.4) * 0.25 + (Math.random() - 0.5) * 0.12);
        }
        drawWaveform(liveCanvasBuffer);
    }
});
