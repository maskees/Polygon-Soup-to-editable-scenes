document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultSection = document.getElementById('result-section');
    
    const progressFill = document.getElementById('progress-fill');
    const statusText = document.getElementById('status-text');
    const viewer = document.getElementById('viewer');
    
    const downloadBtn = document.getElementById('download-btn');
    const resetBtn = document.getElementById('reset-btn');

    let downloadUrl = '';

    // UI State Management
    function showSection(section) {
        uploadSection.classList.remove('active');
        processingSection.classList.remove('active');
        resultSection.classList.remove('active');
        section.classList.add('active');
    }

    // Drag and Drop Events
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Reset Flow
    resetBtn.addEventListener('click', () => {
        fileInput.value = '';
        viewer.src = '';
        downloadUrl = '';
        showSection(uploadSection);
    });

    // Download Output
    downloadBtn.addEventListener('click', () => {
        if (downloadUrl) {
            window.location.href = downloadUrl;
        }
    });

    // Fake Progress Bar (since backend doesn't stream progress)
    function simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            if (progress >= 90) {
                clearInterval(interval);
            } else {
                progress += Math.random() * 5;
                if (progress > 90) progress = 90;
                progressFill.style.width = `${progress}%`;
                
                // Update text based on progress
                if (progress < 20) statusText.innerText = 'Extracting Silhouettes (SAM 2)...';
                else if (progress < 50) statusText.innerText = 'Reconstructing 3D Mesh (CRM)...';
                else if (progress < 80) statusText.innerText = 'Semantic Slicing (SAMPart3D)...';
                else statusText.innerText = 'Assembling OpenUSD Scene...';
            }
        }, 1000);
        return interval;
    }

    // Handle File Upload and API Request
    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (PNG, JPG)');
            return;
        }

        showSection(processingSection);
        const progressInterval = simulateProgress();

        const formData = new FormData();
        formData.append('image', file);

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
            
            // Set download URL
            downloadUrl = data.download_url;
            
            // Note: model-viewer supports USDZ on iOS, but usually GLTF/GLB everywhere else.
            // Since our pipeline outputs USDA, we might need a converter, but we'll try to feed it to model-viewer.
            // If it fails, we just rely on the download button.
            viewer.src = data.download_url;
            
            setTimeout(() => {
                showSection(resultSection);
            }, 1000);

        } catch (error) {
            clearInterval(progressInterval);
            alert(`Error: ${error.message}\n\nNote: The backend may not have the required models downloaded or PyTorch is missing.`);
            showSection(uploadSection);
        }
    }
});
