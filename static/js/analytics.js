document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const loadImgBtn = document.getElementById('loadImgDataBtn');
    const loadVidBtn = document.getElementById('loadVidDataBtn');
    const noDataMsg = document.getElementById('noDataMessage');
    const content = document.getElementById('analyticsContent');
    const summaryCards = document.getElementById('analyticsSummaryCards');
    const lastUpdated = document.getElementById('lastUpdatedTime');
    const annotatedImg = document.getElementById('latestAnnotatedImage');
    const noImgText = document.getElementById('noImageText');
    const trendRow = document.getElementById('trendRow');

    // Chart Instances
    let pieChart = null;
    let lineChart = null;

    // Set defaults for Chart.js dark theme
    Chart.defaults.color = '#e0e0e0';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

    // Initial check for data
    if (sessionStorage.getItem('latestImageData')) {
        loadImgBtn.classList.remove('btn-outline-primary');
        loadImgBtn.classList.add('btn-primary');
    }
    if (sessionStorage.getItem('latestVideoData')) {
        loadVidBtn.classList.remove('btn-outline-info');
        loadVidBtn.classList.add('btn-info');
    }

    // Event Listeners
    loadImgBtn.addEventListener('click', () => {
        const dataStr = sessionStorage.getItem('latestImageData');
        if (dataStr) {
            try {
                const data = JSON.parse(dataStr);
                renderAnalytics(data, 'image');
                updateTimestamp();
            } catch (e) {
                showAlert('Error parsing image data', 'danger');
            }
        } else {
            showAlert('No image analysis data found in session. Please run analysis first.', 'warning');
        }
    });

    loadVidBtn.addEventListener('click', () => {
        const dataStr = sessionStorage.getItem('latestVideoData');
        if (dataStr) {
            try {
                const data = JSON.parse(dataStr);
                renderAnalytics(data, 'video');
                updateTimestamp();
            } catch (e) {
                showAlert('Error parsing video data', 'danger');
            }
        } else {
            showAlert('No video analysis data found in session. Please run analysis first.', 'warning');
        }
    });


    // Main Render Function
    function renderAnalytics(data, type) {
        // Show content, hide empty state
        noDataMsg.classList.add('d-none');
        content.classList.remove('d-none');

        if (type === 'image') {
            renderImageAnalytics(data);
        } else {
            renderVideoAnalytics(data);
        }
    }

    function renderImageAnalytics(data) {
        const stats = data.statistics;
        const detections = data.detections || [];
        const isConfigured = stats.total_slots > 0;

        // 1. Summary Cards
        summaryCards.innerHTML = '';
        if (isConfigured) {
            summaryCards.innerHTML += createStatsCard('Total Slots', stats.total_slots, '#6c757d', '🅿️');
            summaryCards.innerHTML += createStatsCard('Occupied', stats.occupied_slots, '#e63946', '🚗');
            summaryCards.innerHTML += createStatsCard('Available', stats.available_slots, '#2dc653', '✅');
            summaryCards.innerHTML += createStatsCard('Occupancy', formatPercentage(stats.occupancy_rate), '#00b4d8', '📊');
        } else {
            summaryCards.innerHTML += createStatsCard('Total Vehicles', stats.total_vehicles, '#0f3460', '🚙');
        }

        // 2. Vehicle Type Breakdown
        renderPieChart(detections);

        // 3. Annotated Image
        if (data.images && data.images.annotated) {
            annotatedImg.src = 'data:image/jpeg;base64,' + data.images.annotated;
            annotatedImg.style.display = 'block';
            noImgText.style.display = 'none';
        }

        // 4. Hide Trend Chart
        trendRow.style.display = 'none';
    }

    function renderVideoAnalytics(data) {
        const frameData = data.frame_data || [];
        if (frameData.length === 0) return;
        
        const finalFrame = frameData[frameData.length - 1];
        const isConfigured = finalFrame.total_slots > 0;

        // 1. Summary Cards (using last frame)
        summaryCards.innerHTML = '';
        if (isConfigured) {
            summaryCards.innerHTML += createStatsCard('Total Slots', finalFrame.total_slots, '#6c757d', '🅿️');
            summaryCards.innerHTML += createStatsCard('Occupied', finalFrame.occupied, '#e63946', '🚗');
            summaryCards.innerHTML += createStatsCard('Available', finalFrame.available, '#2dc653', '✅');
            summaryCards.innerHTML += createStatsCard('Occupancy', formatPercentage(finalFrame.occupancy_rate), '#00b4d8', '📊');
        } else {
            summaryCards.innerHTML += createStatsCard('Total Vehicles', finalFrame.total_vehicles, '#0f3460', '🚙');
        }

        // 2. Vehicle Breakdown (Simulated based on counts or dummy if not detailed per frame)
        // For video, we might not pass full detection list per frame to save space.
        // Let's create a generic chart if data isn't there.
        renderVideoPieChart(finalFrame);

        // 3. Hide Image
        annotatedImg.style.display = 'none';
        noImgText.style.display = 'block';
        noImgText.textContent = "Check Download link in Video Analysis for full results.";

        // 4. Show Trend Chart
        trendRow.style.display = 'block';
        renderTrendChart(frameData, isConfigured);
    }

    // Chart Renderers
    function renderPieChart(detections) {
        const ctx = document.getElementById('vehiclePieChart').getContext('2d');
        
        if (pieChart) pieChart.destroy();

        // Count classes
        const counts = {};
        detections.forEach(d => {
            counts[d.class] = (counts[d.class] || 0) + 1;
        });

        const labels = Object.keys(counts);
        const data = Object.values(counts);
        
        if (labels.length === 0) {
            labels.push('None');
            data.push(1);
        }

        pieChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#00b4d8', '#e63946', '#2dc653', '#f4a261', '#9b5de5'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' }
                },
                cutout: '70%'
            }
        });
    }

    function renderVideoPieChart(finalFrame) {
        const ctx = document.getElementById('vehiclePieChart').getContext('2d');
        if (pieChart) pieChart.destroy();

        // Simplified data if specific classes aren't available
        pieChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Detected Vehicles'],
                datasets: [{
                    data: [finalFrame.total_vehicles || 1],
                    backgroundColor: ['#00b4d8'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'bottom' } },
                cutout: '70%'
            }
        });
    }

    function renderTrendChart(frameData, isConfigured) {
        const ctx = document.getElementById('trendChart').getContext('2d');
        if (lineChart) lineChart.destroy();

        const labels = frameData.map(d => `F${d.frame}`);
        
        let datasets = [];
        if (isConfigured) {
            datasets = [
                {
                    label: 'Occupied Slots',
                    data: frameData.map(d => d.occupied),
                    borderColor: '#e63946',
                    tension: 0.3
                },
                {
                    label: 'Available Slots',
                    data: frameData.map(d => d.available),
                    borderColor: '#2dc653',
                    tension: 0.3
                }
            ];
        } else {
            datasets = [
                {
                    label: 'Vehicle Count',
                    data: frameData.map(d => d.total_vehicles),
                    borderColor: '#00b4d8',
                    tension: 0.3
                }
            ];
        }

        lineChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { tooltip: { mode: 'index' } }
            }
        });
    }

    function updateTimestamp() {
        const now = new Date();
        lastUpdated.textContent = `Last Updated: ${now.toLocaleTimeString()}`;
    }
});
