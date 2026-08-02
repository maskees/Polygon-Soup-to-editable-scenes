document.addEventListener('DOMContentLoaded', () => {
    // Mode toggles
    const modeRadios = document.querySelectorAll('input[name="upload-mode"]');
    const dropZoneSingle = document.getElementById('drop-zone-single');
    const dropZoneMulti = document.getElementById('drop-zone-multi');
    const fileInputSingle = document.getElementById('file-input-single');
    const fileInputMulti = document.getElementById('file-input-multi');
    const previewSingle = document.getElementById('preview-single');
    const previewMulti = document.getElementById('preview-multi');
    const startBtn = document.getElementById('start-btn');

    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultSection = document.getElementById('result-section');
    
    const progressFill = document.getElementById('progress-fill');
    const statusText = document.getElementById('status-text');
    const viewer = document.getElementById('viewer');
    const downloadBtn = document.getElementById('download-btn');
    const resetBtn = document.getElementById('reset-btn');

    let currentMode = 'single';
    let singleFile = null;
    let multiFiles = [];
    let downloadUrl = '';

    // Switch mode
    modeRadios.forEach(r => {
        r.addEventListener('change', (e) => {
            currentMode = e.target.value;
            if (currentMode === 'single') {
                dropZoneSingle.classList.remove('hidden');
                dropZoneMulti.classList.add('hidden');
            } else {
                dropZoneSingle.classList.add('hidden');
                dropZoneMulti.classList.remove('hidden');
            }
            updateStartBtn();
        });
    });

    // UI State Management
    function showSection(section) {
        uploadSection.classList.remove('active');
        processingSection.classList.remove('active');
        resultSection.classList.remove('active');
        section.classList.add('active');
    }

    function createPreview(file, container) {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        container.appendChild(img);
    }

    function updateStartBtn() {
        if (currentMode === 'single') {
            startBtn.disabled = !singleFile;
        } else {
            startBtn.disabled = multiFiles.length !== 4;
        }
    }

    // Single mode events
    dropZoneSingle.addEventListener('click', () => fileInputSingle.click());
    dropZoneSingle.addEventListener('dragover', (e) => { e.preventDefault(); dropZoneSingle.classList.add('dragover'); });
    dropZoneSingle.addEventListener('dragleave', () => dropZoneSingle.classList.remove('dragover'));
    dropZoneSingle.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZoneSingle.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleSingleFile(e.dataTransfer.files[0]);
    });
    fileInputSingle.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleSingleFile(e.target.files[0]);
    });

    function handleSingleFile(file) {
        if (!file.type.startsWith('image/')) { alert('Please upload an image file'); return; }
        singleFile = file;
        previewSingle.innerHTML = '';
        createPreview(file, previewSingle);
        updateStartBtn();
    }

    // Multi mode events
    dropZoneMulti.addEventListener('click', () => fileInputMulti.click());
    dropZoneMulti.addEventListener('dragover', (e) => { e.preventDefault(); dropZoneMulti.classList.add('dragover'); });
    dropZoneMulti.addEventListener('dragleave', () => dropZoneMulti.classList.remove('dragover'));
    dropZoneMulti.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZoneMulti.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleMultiFiles(e.dataTransfer.files);
    });
    fileInputMulti.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleMultiFiles(e.target.files);
    });

    function handleMultiFiles(files) {
        const validFiles = Array.from(files).filter(f => f.type.startsWith('image/')).slice(0, 4);
        validFiles.forEach(f => {
            if (multiFiles.length < 4) multiFiles.push(f);
        });
        
        previewMulti.innerHTML = '';
        multiFiles.forEach(f => createPreview(f, previewMulti));
        
        if (multiFiles.length > 4) {
            multiFiles = multiFiles.slice(0, 4);
            alert('Only 4 images allowed (Front, Back, Left, Right).');
        }
        updateStartBtn();
    }

    // Reset Flow
    resetBtn.addEventListener('click', () => {
        fileInputSingle.value = '';
        fileInputMulti.value = '';
        singleFile = null;
        multiFiles = [];
        previewSingle.innerHTML = '';
        previewMulti.innerHTML = '';
        viewer.src = '';
        downloadUrl = '';
        updateStartBtn();
        showSection(uploadSection);
    });

    downloadBtn.addEventListener('click', () => {
        if (downloadUrl) window.location.href = downloadUrl;
    });

    function simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            if (progress >= 90) clearInterval(interval);
            else {
                progress += Math.random() * 2;
                if (progress > 90) progress = 90;
                progressFill.style.width = `${progress}%`;
                
                if (progress < 20) statusText.innerText = 'Extracting Silhouettes (SAM 2)...';
                else if (progress < 50) statusText.innerText = currentMode === 'single' ? 'Reconstructing (CRM)...' : 'Reconstructing (Unique3D)...';
                else if (progress < 70) statusText.innerText = 'Generating Quad Topology...';
                else if (progress < 85) statusText.innerText = 'Semantic Slicing (SAMPart3D)...';
                else statusText.innerText = 'Assembling OpenUSD Scene...';
            }
        }, 1000);
        return interval;
    }

    startBtn.addEventListener('click', async () => {
        showSection(processingSection);
        const progressInterval = simulateProgress();

        const formData = new FormData();
        formData.append('mode', currentMode);

        if (currentMode === 'single') {
            formData.append('image', singleFile);
        } else {
            // Front, back, left, right assuming order of upload
            const keys = ['front', 'back', 'left', 'right'];
            multiFiles.forEach((file, index) => {
                formData.append(keys[index], file);
            });
        }

        try {
            const response = await fetch('/api/reconstruct', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            progressFill.style.width = '100%';
            statusText.innerText = 'Finalizing...';

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Pipeline failed');
            }

            const data = await response.json();
            downloadUrl = data.download_url;
            viewer.src = data.download_url;
            
            setTimeout(() => {
                showSection(resultSection);
            }, 1000);

        } catch (error) {
            clearInterval(progressInterval);
            alert(`Error: ${error.message}`);
            showSection(uploadSection);
        }
    });
});
