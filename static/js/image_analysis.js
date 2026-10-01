document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const preprocessBtn = document.getElementById('preprocessBtn');
    const configSelect = document.getElementById('configSelect');
    
    // Sliders
    const confThreshold = document.getElementById('confThreshold');
    const confValue = document.getElementById('confValue');
    const occThreshold = document.getElementById('occThreshold');
    const occValue = document.getElementById('occValue');

    // Sections
    const emptyState = document.getElementById('emptyState');
    const resultsContainer = document.getElementById('resultsContainer');
    const preprocessResults = document.getElementById('preprocessResults');

    // Setup Drop Zone
    setupDropZone('dropZone', 'fileInput', null, (file) => {
        analyzeBtn.disabled = false;
        preprocessBtn.disabled = false;
    });

    // Update Slider Values
    confThreshold.addEventListener('input', (e) => confValue.textContent = e.target.value);
    occThreshold.addEventListener('input', (e) => occValue.textContent = e.target.value);

    // Load available configurations
    loadConfigurations();

    // Helper to ensure clean, standard Data URI
    function formatBase64DataUri(val) {
        if (!val || typeof val !== 'string') return '';
        const trimmed = val.trim();
        if (trimmed.startsWith('data:image/') || trimmed.startsWith('/') || trimmed.startsWith('http')) {
            return trimmed;
        }
        if (trimmed.includes('base64,')) {
            return 'data:image/jpeg;base64,' + trimmed.split('base64,')[1];
        }
        return 'data:image/jpeg;base64,' + trimmed;
    }

    // Preprocessing Handler
    preprocessBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        if (!fileInput.files.length) {
            showAlert('Please select an image file first.', 'warning');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            preprocessBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            preprocessBtn.disabled = true;

            const data = await apiCall('/api/preprocess', 'POST', formData, true);
            
            const origSrc = formatBase64DataUri(data.original || data.original_url);
            const graySrc = formatBase64DataUri(data.gray || data.grayscale);
            const blurSrc = formatBase64DataUri(data.blurred || data.blur);
            const contrastSrc = formatBase64DataUri(data.contrast || data.contrast_enhanced);
            const edgesSrc = formatBase64DataUri(data.edges);

            // Set image sources
            const prepOriginalEl = document.getElementById('prepOriginal');
            const prepGrayEl = document.getElementById('prepGray');
            const prepBlurEl = document.getElementById('prepBlur');
            const prepContrastEl = document.getElementById('prepContrast');
            const prepEdgesEl = document.getElementById('prepEdges');

            if (prepOriginalEl) prepOriginalEl.src = origSrc;
            if (prepGrayEl) prepGrayEl.src = graySrc;
            if (prepBlurEl) prepBlurEl.src = blurSrc;
            if (prepContrastEl) prepContrastEl.src = contrastSrc;
            if (prepEdgesEl) prepEdgesEl.src = edgesSrc;

            preprocessResults.classList.remove('d-none');
            
            // Ensure collapse is open
            const collapseEl = document.getElementById('preprocessCollapse');
            if (collapseEl) {
                try {
                    if (window.bootstrap && bootstrap.Collapse) {
                        const collapseObj = bootstrap.Collapse.getOrCreateInstance(collapseEl);
                        collapseObj.show();
                    } else {
                        collapseEl.classList.add('show');
                    }
                } catch (err) {
                    collapseEl.classList.add('show');
                }
            }
            showAlert('Preprocessing pipeline completed successfully. All 5 stages displayed.', 'success');

        } catch (error) {
            console.error('Preprocessing failed:', error);
            showAlert(error.message || 'Preprocessing failed. Please check the uploaded image.', 'danger');
        } finally {
            preprocessBtn.innerHTML = 'Run Preprocessing';
            preprocessBtn.disabled = false;
        }
    });

    // Main Analysis Handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            showAlert('Please select a parking lot image first.', 'warning');
            return;
        }

        const formData = new FormData(form);

        // UI State Update
        emptyState.classList.add('d-none');
        resultsContainer.classList.add('d-none');
        showLoading('loadingIndicator');
        analyzeBtn.disabled = true;

        try {
            const data = await apiCall('/api/analyze-image', 'POST', formData, true);
            
            if (!data || !data.statistics) {
                throw new Error(data && data.error ? data.error : 'Unexpected response format from detection pipeline.');
            }

            // Store for analytics page
            sessionStorage.setItem('latestImageData', JSON.stringify(data));
            
            renderResults(data);
            
            hideLoading('loadingIndicator');
            resultsContainer.classList.remove('d-none');
            
            if (data.warning) {
                showAlert(data.warning, 'warning');
            } else {
                showAlert('Analysis complete! Parking occupancy computed successfully.', 'success');
            }
        } catch (error) {
            hideLoading('loadingIndicator');
            emptyState.classList.remove('d-none');
            showAlert(error.message || 'Image analysis failed. Please try again.', 'danger');
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    // Functions
    async function loadConfigurations() {
        try {
            const data = await apiCall('/api/list-configs');
            if (data.configs && data.configs.length > 0) {
                data.configs.forEach(config => {
                    const option = document.createElement('option');
                    option.value = config;
                    option.textContent = config;
                    // Auto-select default parking slots configuration if available
                    if (config === 'parking_slots.json') {
                        option.selected = true;
                    }
                    configSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load configs:', error);
        }
    }

    function renderResults(data) {
        const stats = data.statistics;
        
        // 1. Stats Cards
        const statsContainer = document.getElementById('statsCards');
        const isConfigured = stats.total_slots > 0;
        
        statsContainer.innerHTML = '';
        
        if (isConfigured) {
            statsContainer.innerHTML += createStatsCard('Total Slots', stats.total_slots, '#6c757d', '🅿️');
            statsContainer.innerHTML += createStatsCard('Occupied', stats.occupied_slots, '#e63946', '🚗');
            statsContainer.innerHTML += createStatsCard('Available', stats.available_slots, '#2dc653', '✅');
            statsContainer.innerHTML += createStatsCard('Occupancy', formatPercentage(stats.occupancy_rate), '#00b4d8', '📊');
            statsContainer.innerHTML += createStatsCard('Vehicles Detected', stats.total_vehicles, '#f4a261', '🚙');
        } else {
            statsContainer.innerHTML += createStatsCard('Vehicles Detected', stats.total_vehicles, '#0f3460', '🚙');
        }

        // 2. Images
        const origSrc = formatBase64DataUri(data.images ? (data.images.original || data.images.original_url) : '');
        const annotSrc = formatBase64DataUri(data.images ? (data.images.annotated || data.images.annotated_url) : '');

        document.getElementById('resultOriginalImg').src = origSrc;
        document.getElementById('resultAnnotatedImg').src = annotSrc;
        
        // Setup Download button
        const downloadBtn = document.getElementById('downloadResultBtn');
        downloadBtn.href = annotSrc;

        // 3. Slot Table
        const slotTbody = document.getElementById('slotTableBody');
        slotTbody.innerHTML = '';
        
        if (isConfigured && data.slot_status && data.slot_status.length > 0) {
            data.slot_status.forEach(slot => {
                const isOcc = slot.status === 'Occupied';
                const badgeClass = isOcc ? 'status-occupied' : 'status-available';
                const tr = `
                    <tr>
                        <td><strong>${slot.id}</strong></td>
                        <td>${slot.label || '-'}</td>
                        <td><span class="badge ${badgeClass}">${slot.status}</span></td>
                    </tr>
                `;
                slotTbody.innerHTML += tr;
            });
        } else {
            slotTbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">No parking slots configured for this analysis</td></tr>';
        }

        // 4. Detection Table
        const detTbody = document.getElementById('detectionTableBody');
        detTbody.innerHTML = '';
        
        if (data.detections && data.detections.length > 0) {
            data.detections.forEach(det => {
                const tr = `
                    <tr>
                        <td class="text-capitalize"><strong>${det.class}</strong></td>
                        <td>${formatPercentage(det.confidence)}</td>
                        <td><small>[${det.bbox.map(n => Math.round(n)).join(', ')}]</small></td>
                    </tr>
                `;
                detTbody.innerHTML += tr;
            });
        } else {
            detTbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">No vehicles detected in image</td></tr>';
        }
    }
});
