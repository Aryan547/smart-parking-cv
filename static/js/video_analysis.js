document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const configSelect = document.getElementById('configSelect');
    
    // Sliders
    const confThreshold = document.getElementById('confThreshold');
    const confValue = document.getElementById('confValue');
    const occThreshold = document.getElementById('occThreshold');
    const occValue = document.getElementById('occValue');

    // Sections
    const emptyState = document.getElementById('emptyState');
    const resultsContainer = document.getElementById('resultsContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    let chartInstance = null;

    // Setup Drop Zone
    setupDropZone('dropZone', 'fileInput', null, (file) => {
        analyzeBtn.disabled = false;
    });

    // Update Slider Values
    confThreshold.addEventListener('input', (e) => confValue.textContent = e.target.value);
    occThreshold.addEventListener('input', (e) => occValue.textContent = e.target.value);

    // Load available configurations
    loadConfigurations();

    // Main Analysis Handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            showAlert('Please select a video first.', 'warning');
            return;
        }

        const formData = new FormData(form);

        // UI State Update
        emptyState.classList.add('d-none');
        resultsContainer.classList.add('d-none');
        showLoading('loadingIndicator');
        analyzeBtn.disabled = true;
        
        // Simulating progress bar for UI UX since actual server SSE might be complex
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 5;
            if(progress > 90) progress = 90; // cap at 90 until done
            updateProgress(progress, 'Processing frames...');
        }, 1000);

        try {
            // Note: For large videos, a real app would use WebSockets or SSE for progress.
            // Using a standard HTTP request here for simplicity, expecting a longer response time.
            const data = await apiCall('/api/analyze-video', 'POST', formData, true);
            
            clearInterval(progressInterval);
            updateProgress(100, 'Finalizing...');
            
            // Store for analytics page
            sessionStorage.setItem('latestVideoData', JSON.stringify(data));
            
            renderResults(data);
            
            setTimeout(() => {
                hideLoading('loadingIndicator');
                resultsContainer.classList.remove('d-none');
                showAlert('Video analysis complete!', 'success');
            }, 500);

        } catch (error) {
            clearInterval(progressInterval);
            hideLoading('loadingIndicator');
            emptyState.classList.remove('d-none');
            // Error handled by apiCall
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function updateProgress(percent, text) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = `${Math.round(percent)}%`;
        if(text) progressText.textContent = text;
    }

    async function loadConfigurations() {
        try {
            const data = await apiCall('/api/list-configs');
            if (data.configs && data.configs.length > 0) {
                data.configs.forEach(config => {
                    const option = document.createElement('option');
                    option.value = config;
                    option.textContent = config;
                    configSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load configs:', error);
        }
    }

    function renderResults(data) {
        // Setup Download Link
        if(data.output_video_url) {
            const dlBtn = document.getElementById('downloadVideoBtn');
            dlBtn.href = data.output_video_url;
            dlBtn.download = 'annotated_video.mp4';
        }

        // Stats Cards (Using last frame data)
        const statsContainer = document.getElementById('statsCards');
        const frameData = data.frame_data;
        
        if (frameData && frameData.length > 0) {
            const finalFrame = frameData[frameData.length - 1];
            const isConfigured = finalFrame.total_slots > 0;
            
            statsContainer.innerHTML = '';
            
            if (isConfigured) {
                statsContainer.innerHTML += createStatsCard('Total Slots', finalFrame.total_slots, '#6c757d', '🅿️');
                statsContainer.innerHTML += createStatsCard('Occupied', finalFrame.occupied, '#e63946', '🚗');
                statsContainer.innerHTML += createStatsCard('Available', finalFrame.available, '#2dc653', '✅');
                statsContainer.innerHTML += createStatsCard('Occupancy', formatPercentage(finalFrame.occupancy_rate), '#00b4d8', '📊');
            } else {
                statsContainer.innerHTML += createStatsCard('Total Vehicles', finalFrame.total_vehicles, '#0f3460', '🚙');
            }

            // Render Chart
            renderChart(frameData, isConfigured);
            
            // Render Table
            renderTable(frameData, isConfigured);
        }
    }

    function renderChart(frameData, isConfigured) {
        const ctx = document.getElementById('occupancyChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        const labels = frameData.map(d => `Frame ${d.frame}`);
        
        let datasets = [];
        if (isConfigured) {
            datasets = [
                {
                    label: 'Occupancy Rate (%)',
                    data: frameData.map(d => Math.round(d.occupancy_rate * 100)),
                    borderColor: '#00b4d8',
                    backgroundColor: 'rgba(0, 180, 216, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }
            ];
        } else {
            datasets = [
                {
                    label: 'Total Vehicles',
                    data: frameData.map(d => d.total_vehicles),
                    borderColor: '#f4a261',
                    backgroundColor: 'rgba(244, 162, 97, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }
            ];
        }

        Chart.defaults.color = '#9ca3af';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: isConfigured ? 100 : undefined
                    }
                }
            }
        });
    }

    function renderTable(frameData, isConfigured) {
        const tbody = document.getElementById('frameTableBody');
        tbody.innerHTML = '';
        
        // Reverse array to show newest first if desired, or keep sequential
        frameData.forEach(d => {
            const tr = `
                <tr>
                    <td>${d.frame}</td>
                    <td>${d.total_slots || '-'}</td>
                    <td>${d.occupied !== undefined ? d.occupied : '-'}</td>
                    <td>${d.available !== undefined ? d.available : '-'}</td>
                    <td>${d.occupancy_rate !== undefined ? formatPercentage(d.occupancy_rate) : '-'}</td>
                    <td>${d.total_vehicles}</td>
                </tr>
            `;
            tbody.innerHTML += tr;
        });
    }
});
