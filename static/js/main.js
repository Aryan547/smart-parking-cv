/**
 * Shared Utility Functions for ParkVision AI
 */

// Show loading overlay within a container
function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.classList.remove('d-none');
}

// Hide loading overlay
function hideLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.classList.add('d-none');
}

// Display Bootstrap alert
function showAlert(message, type = 'info') {
    const container = document.getElementById('alert-container');
    if (!container) return;

    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show shadow-sm" role="alert">
            ${message}
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    container.innerHTML = alertHtml;
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Auto dismiss after 6 seconds
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) {
            try {
                const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                bsAlert.close();
            } catch (e) {
                alert.remove();
            }
        }
    }, 6000);
}

// Format number as percentage
function formatPercentage(value) {
    if (value === undefined || value === null || isNaN(value)) return '0.0%';
    return `${(Number(value) * 100).toFixed(1)}%`;
}

// Simple number formatting
function formatNumber(value) {
    if (value === undefined || value === null || isNaN(value)) return '0';
    return value.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Generate HTML for a stats card
function createStatsCard(label, value, colorHex, iconStr = '') {
    return `
        <div class="col-md-3 col-sm-6">
            <div class="stat-card shadow-sm" style="border-left: 4px solid ${colorHex};">
                <div class="stat-icon" style="color: ${colorHex};">${iconStr}</div>
                <div>
                    <div class="stat-label">${label}</div>
                    <div class="stat-value text-light">${value}</div>
                </div>
            </div>
        </div>
    `;
}

// API Call Wrapper
async function apiCall(url, method = 'GET', data = null, isFormData = false) {
    const options = {
        method: method,
    };

    if (data) {
        if (isFormData) {
            options.body = data; // Fetch automatically sets multipart/form-data with boundary
        } else {
            options.headers = {
                'Content-Type': 'application/json'
            };
            options.body = JSON.stringify(data);
        }
    }

    try {
        const response = await fetch(url, options);
        let result;
        const contentType = response.headers.get("content-type");
        
        if (contentType && contentType.includes("application/json")) {
            result = await response.json();
        } else {
            const text = await response.text();
            throw new Error(text || `Server returned status ${response.status}`);
        }
        
        if (!response.ok || (result && result.success === false)) {
            const errorMsg = (result && (result.error || result.message)) || `Server returned error (${response.status})`;
            throw new Error(errorMsg);
        }
        
        return result;
    } catch (error) {
        console.error('API Call Error:', error);
        showAlert(error.message, 'danger');
        throw error;
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Setup File Drop Zone logic
function setupDropZone(dropZoneId, inputId, previewSelector = null, callback = null) {
    const dropZoneElement = document.getElementById(dropZoneId);
    const inputElement = document.getElementById(inputId);
    
    if (!dropZoneElement || !inputElement) return;

    // Click to open file dialog
    dropZoneElement.addEventListener('click', (e) => {
        // Prevent clicking if clicking on the input itself
        if(e.target !== inputElement) {
            inputElement.click();
        }
    });

    // Handle drag over
    dropZoneElement.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZoneElement.classList.add('drop-zone--over');
    });

    // Handle drag leave
    ['dragleave', 'dragend'].forEach(type => {
        dropZoneElement.addEventListener(type, (e) => {
            dropZoneElement.classList.remove('drop-zone--over');
        });
    });

    // Handle drop
    dropZoneElement.addEventListener('drop', (e) => {
        e.preventDefault();
        
        if (e.dataTransfer.files.length) {
            inputElement.files = e.dataTransfer.files;
            updateThumbnail(dropZoneElement, e.dataTransfer.files[0], previewSelector);
            if (callback) callback(e.dataTransfer.files[0]);
        }
        
        dropZoneElement.classList.remove('drop-zone--over');
    });

    // Handle input change
    inputElement.addEventListener('change', (e) => {
        if (inputElement.files.length) {
            updateThumbnail(dropZoneElement, inputElement.files[0], previewSelector);
            if (callback) callback(inputElement.files[0]);
        }
    });
}

// Update Drop Zone Thumbnail
function updateThumbnail(dropZoneElement, file, previewSelector) {
    let thumbnailElement = dropZoneElement.querySelector('.drop-zone-thumb');
    let promptElement = dropZoneElement.querySelector('.drop-zone-prompt');
    let nameElement = dropZoneElement.querySelector('.drop-zone-file-name');

    // Hide prompt
    if (promptElement) promptElement.style.display = 'none';

    // Show thumbnail for images
    if (file.type.startsWith('image/')) {
        if (thumbnailElement) {
            thumbnailElement.style.display = 'block';
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                thumbnailElement.style.backgroundImage = `url('${reader.result}')`;
                if(previewSelector) {
                    const extraPreview = document.querySelector(previewSelector);
                    if(extraPreview) extraPreview.src = reader.result;
                }
            };
        }
    } 
    // Show name for videos
    else if (file.type.startsWith('video/')) {
        if (thumbnailElement) thumbnailElement.style.display = 'none';
        if (nameElement) {
            nameElement.textContent = file.name;
            nameElement.style.display = 'block';
        }
    }
}
