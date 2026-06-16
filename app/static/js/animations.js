// Detection page workflow animations and API calls

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('detectionForm');
    const inputSection = document.getElementById('inputSection');
    const animationSection = document.getElementById('animationSection');
    const resultSection = document.getElementById('resultSection');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    
    // Animation Elements
    const animStepTitle = document.getElementById('animStepTitle');
    const animStepDesc = document.getElementById('animStepDesc');
    const animProgressBar = document.getElementById('animProgressBar');
    
    if(!form) return; // Only run on detect page
    
    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const content = document.getElementById('newsContent').value;
        
        if(!content) return;
        
        // 1. Hide input, show animation
        inputSection.classList.add('d-none');
        animationSection.classList.remove('d-none');
        resultSection.classList.add('d-none');
        
        // Start multi-step animation
        await runAnimationSequence();
        
        // 2. Fetch Prediction
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content })
            });
            
            const data = await response.json();
            
            if(data.error) {
                alert("Error: " + data.error);
                resetUI();
                return;
            }
            
            // 3. Populate Result UI
            populateResultUI(data);
            
            // 4. Show Result Section
            animationSection.classList.add('d-none');
            resultSection.classList.remove('d-none');
            
            // Animate confidence bar
            setTimeout(() => {
                document.getElementById('confidenceBar').style.width = data.confidence + '%';
            }, 500);
            
        } catch (err) {
            console.error("Fetch error:", err);
            alert("Failed to connect to the prediction server.");
            resetUI();
        }
    });
    
    // Reset buttons
    clearBtn.addEventListener('click', () => {
        form.reset();
    });
    
    analyzeAnotherBtn.addEventListener('click', () => {
        resetUI();
    });
    
    function resetUI() {
        form.reset();
        inputSection.classList.remove('d-none');
        animationSection.classList.add('d-none');
        resultSection.classList.add('d-none');
        document.getElementById('confidenceBar').style.width = '0%';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    async function runAnimationSequence() {
        const steps = [
            { title: "Initializing Detection Engine...", desc: "Loading neural network nodes...", duration: 1500, progress: 15 },
            { title: "Scanning News Content...", desc: "Extracting linguistic patterns and context...", duration: 2500, progress: 40 },
            { title: "Extracting Linguistic Features...", desc: "Tokenization and TF-IDF extraction...", duration: 1500, progress: 60 },
            { title: "Running Machine Learning Models...", desc: "Processing via DistilBERT...", duration: 2500, progress: 85 },
            { title: "Verifying News Authenticity...", desc: "Cross-referencing parameters...", duration: 1500, progress: 95 },
            { title: "Calculating Confidence Score...", desc: "Finalizing probability matrix...", duration: 1000, progress: 100 }
        ];
        
        for(let step of steps) {
            animStepTitle.innerText = step.title;
            animStepDesc.innerText = step.desc;
            animProgressBar.style.width = step.progress + '%';
            await sleep(step.duration);
        }
    }
    
    function populateResultUI(data) {
        const finalCard = document.getElementById('finalResultCard');
        const iconDiv = document.getElementById('resultIcon');
        const textH1 = document.getElementById('resultText');
        const riskText = document.getElementById('riskText');
        
        // Reset classes
        finalCard.className = "glass-card p-5 mb-4 text-center result-card-anim";
        document.getElementById('confidenceBar').className = "progress-bar";
        
        if(data.result === 'REAL') {
            finalCard.classList.add('border', 'border-success', 'border-opacity-50');
            finalCard.style.boxShadow = "0 0 30px rgba(34, 197, 94, 0.2)";
            
            iconDiv.innerHTML = '<i class="fa-solid fa-circle-check fa-5x text-success pulse-icon"></i>';
            textH1.innerHTML = '<span class="text-success">REAL NEWS</span>';
            riskText.innerHTML = '<span class="text-success"><i class="fa-solid fa-shield-check me-2"></i>Verified Authentic</span>';
            
            document.getElementById('confidenceBar').classList.add('bg-success');
        } else {
            finalCard.classList.add('border', 'border-danger', 'border-opacity-50');
            finalCard.style.boxShadow = "0 0 30px rgba(239, 68, 68, 0.3)";
            
            iconDiv.innerHTML = '<i class="fa-solid fa-triangle-exclamation fa-5x text-danger pulse-icon" style="animation-duration:1s"></i>';
            textH1.innerHTML = '<span class="text-danger">FAKE NEWS</span>';
            riskText.innerHTML = `<span class="text-danger"><i class="fa-solid fa-radiation me-2"></i>High Risk (${data.confidence}% Fake Probability)</span>`;
            
            document.getElementById('confidenceBar').classList.add('bg-danger');
        }
        
        // Stats
        animateValue(document.getElementById('confidenceValueText'), 0, data.confidence, 1000);
        document.getElementById('modelUsedText').innerText = data.model_used;
        document.getElementById('processingTimeText').innerText = data.processing_time + 's';
        
        // Keywords
        const negContainer = document.getElementById('negativeFactors');
        const posContainer = document.getElementById('positiveFactors');
        
        negContainer.innerHTML = '';
        posContainer.innerHTML = '';
        
        if(data.top_negative && data.top_negative.length > 0) {
            data.top_negative.forEach(w => {
                negContainer.innerHTML += `<span class="bg-danger bg-opacity-10 text-danger border-danger"><i class="fa-solid fa-xmark me-1 small"></i>${w}</span>`;
            });
        } else {
            negContainer.innerHTML = '<span class="text-muted border-0">No significant factors found</span>';
        }
        
        if(data.top_positive && data.top_positive.length > 0) {
            data.top_positive.forEach(w => {
                posContainer.innerHTML += `<span class="bg-success bg-opacity-10 text-success border-success"><i class="fa-solid fa-check me-1 small"></i>${w}</span>`;
            });
        } else {
            posContainer.innerHTML = '<span class="text-muted border-0">No significant factors found</span>';
        }
    }
    
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(2) + '%';
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
