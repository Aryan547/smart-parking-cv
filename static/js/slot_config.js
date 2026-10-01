document.addEventListener('DOMContentLoaded', () => {
    // Canvas & Context
    const container = document.getElementById('canvasContainer');
    const canvas = document.getElementById('editorCanvas');
    const ctx = canvas.getContext('2d');
    
    // UI Elements
    const imageInput = document.getElementById('imageInput');
    const newPolygonBtn = document.getElementById('newPolygonBtn');
    const undoBtn = document.getElementById('undoBtn');
    const deleteBtn = document.getElementById('deleteBtn');
    const clearAllBtn = document.getElementById('clearAllBtn');
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    const configNameInput = document.getElementById('configName');
    const loadConfigSelect = document.getElementById('loadConfigSelect');
    const loadConfigBtn = document.getElementById('loadConfigBtn');
    const statusText = document.getElementById('statusText');
    const slotList = document.getElementById('slotList');
    const canvasPrompt = document.getElementById('canvasPrompt');
    const slotCountBadge = document.getElementById('slotCountBadge');

    // Modal
    const labelModal = new bootstrap.Modal(document.getElementById('labelModal'));
    const slotLabelInput = document.getElementById('slotLabelInput');
    const saveLabelBtn = document.getElementById('saveLabelBtn');
    const cancelLabelBtn = document.getElementById('cancelLabelBtn');

    // State Variables
    let img = new Image();
    let imageLoaded = false;
    let originalWidth = 0;
    let originalHeight = 0;
    let scale = 1;
    
    let slots = []; // Array of {id, points: [{x,y}], label}
    let currentPolygon = []; // Array of {x,y}
    let isDrawing = false;
    let selectedSlotId = null;
    let nextSlotId = 1;

    // Load existing configurations list
    loadConfigList();

    // Resize canvas handling
    window.addEventListener('resize', debounce(draw, 100));

    // --- Image Loading ---
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            img.onload = () => {
                imageLoaded = true;
                originalWidth = img.width;
                originalHeight = img.height;
                
                canvasPrompt.style.display = 'none';
                canvas.style.display = 'block';
                
                // Enable buttons
                newPolygonBtn.disabled = false;
                clearAllBtn.disabled = false;
                
                // Clear existing slots on new image upload
                slots = [];
                currentPolygon = [];
                isDrawing = false;
                selectedSlotId = null;
                nextSlotId = 1;
                
                updateSlotList();
                setMode('idle');
                draw();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    // --- Drawing Interaction ---
    newPolygonBtn.addEventListener('click', () => {
        if (!imageLoaded) return;
        setMode('drawing');
        currentPolygon = [];
        selectedSlotId = null;
        draw();
    });

    undoBtn.addEventListener('click', () => {
        if (isDrawing && currentPolygon.length > 0) {
            currentPolygon.pop();
            draw();
            if (currentPolygon.length === 0) {
                setMode('idle');
            }
        }
    });

    canvas.addEventListener('click', (e) => {
        if (!imageLoaded) return;

        const rect = canvas.getBoundingClientRect();
        // Calculate coordinates relative to original image size
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;

        if (isDrawing) {
            // Check if clicking near start point to close
            if (currentPolygon.length >= 3) {
                const startPoint = currentPolygon[0];
                const dist = Math.hypot(startPoint.x - x, startPoint.y - y);
                // 10 pixels threshold (scaled)
                if (dist < 15 / scale) {
                    finishPolygon();
                    return;
                }
            }
            
            currentPolygon.push({x, y});
            draw();
        } else {
            // Check if clicking inside an existing polygon to select it
            const clickedSlot = findClickedSlot(x, y);
            if (clickedSlot) {
                selectedSlotId = clickedSlot.id;
                deleteBtn.disabled = false;
            } else {
                selectedSlotId = null;
                deleteBtn.disabled = true;
            }
            updateSlotListSelection();
            draw();
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing || currentPolygon.length === 0) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;
        
        draw(x, y); // Draw with preview line
    });

    // Press Enter to close polygon
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && isDrawing && currentPolygon.length >= 3) {
            finishPolygon();
        }
        if (e.key === 'Escape' && isDrawing) {
            setMode('idle');
            currentPolygon = [];
            draw();
        }
    });

    function finishPolygon() {
        setMode('modal');
        slotLabelInput.value = `P${nextSlotId}`;
        labelModal.show();
        // Focus input after modal shown
        setTimeout(() => slotLabelInput.focus(), 500);
    }

    // Modal Actions
    saveLabelBtn.addEventListener('click', () => {
        const label = slotLabelInput.value.trim() || `P${nextSlotId}`;
        
        slots.push({
            id: nextSlotId++,
            label: label,
            points: [...currentPolygon]
        });
        
        currentPolygon = [];
        labelModal.hide();
        setMode('idle');
        updateSlotList();
        draw();
        
        saveConfigBtn.disabled = slots.length === 0;
    });

    cancelLabelBtn.addEventListener('click', () => {
        currentPolygon = [];
        labelModal.hide();
        setMode('idle');
        draw();
    });

    // --- Toolbar Actions ---
    deleteBtn.addEventListener('click', () => {
        if (selectedSlotId) {
            slots = slots.filter(s => s.id !== selectedSlotId);
            selectedSlotId = null;
            deleteBtn.disabled = true;
            updateSlotList();
            draw();
            saveConfigBtn.disabled = slots.length === 0;
        }
    });

    clearAllBtn.addEventListener('click', () => {
        if(confirm('Are you sure you want to clear all slots?')) {
            slots = [];
            selectedSlotId = null;
            deleteBtn.disabled = true;
            saveConfigBtn.disabled = true;
            nextSlotId = 1;
            updateSlotList();
            draw();
        }
    });

    // --- Save/Load ---
    saveConfigBtn.addEventListener('click', async () => {
        const name = configNameInput.value.trim();
        if (!name) {
            showAlert('Please enter a configuration name', 'warning');
            configNameInput.focus();
            return;
        }
        
        if (slots.length === 0) return;

        // Format data for backend backend
        const configData = {
            name: name,
            slots: slots.map(s => ({
                id: s.id,
                label: s.label,
                points: s.points.map(p => [p.x, p.y])
            }))
        };

        try {
            saveConfigBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
            saveConfigBtn.disabled = true;
            
            await apiCall('/api/save-slots', 'POST', configData);
            
            showAlert('Configuration saved successfully!', 'success');
            loadConfigList(); // Refresh list
        } catch (error) {
            // error handled by apiCall
        } finally {
            saveConfigBtn.innerHTML = 'Save Configuration';
            saveConfigBtn.disabled = false;
        }
    });

    loadConfigBtn.addEventListener('click', async () => {
        const filename = loadConfigSelect.value;
        if (!filename) return;

        try {
            loadConfigBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
            const data = await apiCall(`/api/load-slots/${filename}`);
            
            // Reconstruct slots array
            slots = data.slots.map(s => {
                // Convert array of [x,y] to object {x,y}
                const pts = Array.isArray(s.points[0]) ? 
                    s.points.map(p => ({x: p[0], y: p[1]})) : 
                    s.points; // Fallback if already objects
                    
                return {
                    id: s.id,
                    label: s.label || `P${s.id}`,
                    points: pts
                };
            });
            
            // Update nextSlotId
            if (slots.length > 0) {
                const maxId = Math.max(...slots.map(s => s.id));
                nextSlotId = maxId + 1;
            } else {
                nextSlotId = 1;
            }
            
            configNameInput.value = data.name || filename.replace('.json', '');
            
            selectedSlotId = null;
            updateSlotList();
            
            // Note: We can draw the polygons even without the image, but it's better with it.
            // If image is loaded, just draw.
            if (imageLoaded) {
                draw();
                showAlert(`Loaded configuration: ${slots.length} slots`, 'success');
            } else {
                showAlert(`Configuration loaded. Please upload the matching image to visualize.`, 'info');
            }
            
            saveConfigBtn.disabled = false;
            clearAllBtn.disabled = false;

        } catch(error) {
            // error handled by apiCall
        } finally {
            loadConfigBtn.innerHTML = 'Load';
        }
    });

    async function loadConfigList() {
        try {
            const data = await apiCall('/api/list-configs');
            loadConfigSelect.innerHTML = '<option value="">-- Load Config --</option>';
            if (data.configs) {
                data.configs.forEach(conf => {
                    const opt = document.createElement('option');
                    opt.value = conf;
                    opt.textContent = conf;
                    loadConfigSelect.appendChild(opt);
                });
            }
        } catch (e) {
            console.error("Failed to load config list", e);
        }
    }


    // --- Canvas Drawing & Logic ---
    function draw(mouseX = null, mouseY = null) {
        if (!imageLoaded) return;

        // Calculate scaling to fit container while maintaining aspect ratio
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        
        const scaleX = containerWidth / originalWidth;
        const scaleY = containerHeight / originalHeight;
        scale = Math.min(scaleX, scaleY, 1); // Don't scale up past original size
        
        // Ensure scale is not too small or 0
        if(scale <= 0) scale = 1;

        const drawWidth = originalWidth * scale;
        const drawHeight = originalHeight * scale;

        canvas.width = drawWidth;
        canvas.height = drawHeight;

        // Clear and draw image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, drawWidth, drawHeight);

        // Draw existing slots
        slots.forEach(slot => {
            const isSelected = slot.id === selectedSlotId;
            drawPolygon(slot.points, isSelected, slot.label);
        });

        // Draw current polygon in progress
        if (isDrawing && currentPolygon.length > 0) {
            ctx.beginPath();
            ctx.moveTo(currentPolygon[0].x * scale, currentPolygon[0].y * scale);
            
            for (let i = 1; i < currentPolygon.length; i++) {
                ctx.lineTo(currentPolygon[i].x * scale, currentPolygon[i].y * scale);
            }
            
            // Draw preview line to mouse
            if (mouseX !== null && mouseY !== null) {
                ctx.lineTo(mouseX * scale, mouseY * scale);
            }
            
            ctx.strokeStyle = '#00b4d8';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw points
            currentPolygon.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x * scale, p.y * scale, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#00b4d8';
                ctx.fill();
            });
            
            // Draw close hint if near start
            if (currentPolygon.length >= 3 && mouseX !== null) {
                const start = currentPolygon[0];
                const dist = Math.hypot(start.x - mouseX, start.y - mouseY);
                if (dist < 15 / scale) {
                    ctx.beginPath();
                    ctx.arc(start.x * scale, start.y * scale, 8, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(0, 180, 216, 0.5)';
                    ctx.fill();
                }
            }
        }
    }

    function drawPolygon(points, isSelected, label) {
        if (!points || points.length < 3) return;

        ctx.beginPath();
        ctx.moveTo(points[0].x * scale, points[0].y * scale);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x * scale, points[i].y * scale);
        }
        ctx.closePath();

        // Fill
        ctx.fillStyle = isSelected ? 'rgba(0, 180, 216, 0.4)' : 'rgba(45, 198, 83, 0.2)';
        ctx.fill();

        // Stroke
        ctx.strokeStyle = isSelected ? '#00b4d8' : '#2dc653';
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.stroke();

        // Label
        if (label) {
            // Find center
            let cx = 0, cy = 0;
            points.forEach(p => { cx += p.x; cy += p.y; });
            cx = (cx / points.length) * scale;
            cy = (cy / points.length) * scale;

            ctx.fillStyle = isSelected ? '#00b4d8' : '#2dc653';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // Text bg
            const metrics = ctx.measureText(label);
            const bgW = metrics.width + 10;
            const bgH = 20;
            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            ctx.fillRect(cx - bgW/2, cy - bgH/2, bgW, bgH);
            
            // Text
            ctx.fillStyle = '#fff';
            ctx.fillText(label, cx, cy);
        }
    }

    // Point in polygon ray-casting algorithm
    function findClickedSlot(px, py) {
        // Iterate backwards to select top-most drawn
        for (let i = slots.length - 1; i >= 0; i--) {
            const points = slots[i].points;
            let inside = false;
            for (let j = 0, k = points.length - 1; j < points.length; k = j++) {
                const xi = points[j].x, yi = points[j].y;
                const xj = points[k].x, yj = points[k].y;
                const intersect = ((yi > py) != (yj > py))
                    && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
                if (intersect) inside = !inside;
            }
            if (inside) return slots[i];
        }
        return null;
    }

    // --- UI Helpers ---
    function setMode(mode) {
        isDrawing = mode === 'drawing';
        undoBtn.disabled = !isDrawing;
        newPolygonBtn.disabled = isDrawing;
        
        if (mode === 'drawing') {
            statusText.textContent = 'Drawing: Click on image to add points. Press Enter to finish.';
            statusText.className = 'status-text text-warning fw-bold';
        } else if (mode === 'idle') {
            statusText.textContent = 'Idle: Select a slot or create a new one.';
            statusText.className = 'status-text text-accent fw-bold';
        } else if (mode === 'modal') {
            statusText.textContent = 'Waiting for input...';
        }
    }

    function updateSlotList() {
        slotCountBadge.textContent = `${slots.length} Slots`;
        
        if (slots.length === 0) {
            slotList.innerHTML = '<li class="list-group-item bg-transparent text-muted text-center py-3">No slots configured</li>';
            return;
        }

        slotList.innerHTML = '';
        slots.forEach(slot => {
            const li = document.createElement('li');
            li.className = `list-group-item bg-transparent text-light border-secondary d-flex justify-content-between align-items-center ${slot.id === selectedSlotId ? 'bg-primary bg-opacity-25' : ''}`;
            li.style.cursor = 'pointer';
            
            li.innerHTML = `
                <div>
                    <span class="badge bg-secondary me-2">ID: ${slot.id}</span>
                    <strong>${slot.label}</strong>
                </div>
                <small class="text-muted">${slot.points.length} pts</small>
            `;
            
            li.addEventListener('click', () => {
                selectedSlotId = slot.id;
                deleteBtn.disabled = false;
                updateSlotListSelection();
                draw();
            });
            
            slotList.appendChild(li);
        });
    }

    function updateSlotListSelection() {
        const items = slotList.querySelectorAll('li');
        items.forEach((item, index) => {
            if (slots[index] && slots[index].id === selectedSlotId) {
                item.classList.add('bg-primary', 'bg-opacity-25');
            } else {
                item.classList.remove('bg-primary', 'bg-opacity-25');
            }
        });
    }
});
