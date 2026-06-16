// Main JavaScript for general interactivity and charts

// Simple Particle Background Generator
function createParticles() {
    const bg = document.getElementById('particles-bg');
    if (!bg) return;
    
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        
        // Random properties
        const size = Math.random() * 5 + 2;
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        const delay = Math.random() * 10;
        const duration = Math.random() * 10 + 10;
        const opacity = Math.random() * 0.5 + 0.1;
        
        // Apply styles
        particle.style.position = 'absolute';
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.background = 'rgba(59, 130, 246, 1)';
        particle.style.borderRadius = '50%';
        particle.style.left = `${posX}vw`;
        particle.style.top = `${posY}vh`;
        particle.style.opacity = opacity;
        particle.style.boxShadow = `0 0 ${size*2}px rgba(59, 130, 246, 0.8)`;
        
        // Animation
        particle.animate([
            { transform: 'translate(0, 0)', opacity: opacity },
            { transform: `translate(${Math.random()*100-50}px, ${-Math.random()*200-100}px)`, opacity: 0 }
        ], {
            duration: duration * 1000,
            delay: delay * 1000,
            iterations: Infinity,
            easing: 'linear'
        });
        
        bg.appendChild(particle);
    }
}

// Animate Counters
function animateCounters() {
    const counters = document.querySelectorAll('.counter');
    const speed = 200; // lower is slower
    
    counters.forEach(counter => {
        const target = +counter.getAttribute('data-target');
        const updateCount = () => {
            const count = +counter.innerText.replace(/,/g, '');
            const inc = target / speed;
            
            if (count < target) {
                counter.innerText = Math.ceil(count + inc).toLocaleString();
                setTimeout(updateCount, 10);
            } else {
                counter.innerText = target.toLocaleString();
            }
        };
        
        // Intersection Observer to start animation when in view
        const observer = new IntersectionObserver((entries) => {
            if(entries[0].isIntersecting) {
                updateCount();
                observer.disconnect();
            }
        });
        observer.observe(counter);
    });
}

// Initialize Charts for Insights Page
function initCharts() {
    fetch('/api/notebook_stats')
        .then(res => res.json())
        .then(data => {
            // Dataset Chart
            if(document.getElementById('datasetChart')) {
                const ctxDataset = document.getElementById('datasetChart').getContext('2d');
                new Chart(ctxDataset, {
                    type: 'doughnut',
                    data: {
                        labels: ['Real News', 'Fake News'],
                        datasets: [{
                            data: [data.dataset_info.real_news, data.dataset_info.fake_news],
                            backgroundColor: ['rgba(34, 197, 94, 0.8)', 'rgba(239, 68, 68, 0.8)'],
                            borderColor: ['#22C55E', '#EF4444'],
                            borderWidth: 1,
                            hoverOffset: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom', labels: { color: '#F8FAFC' } }
                        }
                    }
                });
            }
            
            // Performance Chart
            if(document.getElementById('performanceChart')) {
                const ctxPerf = document.getElementById('performanceChart').getContext('2d');
                new Chart(ctxPerf, {
                    type: 'bar',
                    data: {
                        labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                        datasets: [{
                            label: 'DistilBERT Performance (%)',
                            data: [
                                data.model_performance.accuracy,
                                data.model_performance.precision,
                                data.model_performance.recall,
                                data.model_performance.f1_score,
                                data.model_performance.roc_auc
                            ],
                            backgroundColor: [
                                'rgba(59, 130, 246, 0.7)',
                                'rgba(14, 165, 233, 0.7)',
                                'rgba(34, 197, 94, 0.7)',
                                'rgba(245, 158, 11, 0.7)',
                                'rgba(139, 92, 246, 0.7)'
                            ],
                            borderWidth: 1,
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { 
                                beginAtZero: false, 
                                min: 90, 
                                max: 100,
                                ticks: { color: '#94A3B8' },
                                grid: { color: 'rgba(255,255,255,0.05)' }
                            },
                            x: {
                                ticks: { color: '#94A3B8' },
                                grid: { display: false }
                            }
                        },
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }

            // Confusion Matrix Chart
            if(document.getElementById('confusionMatrixChart') && data.confusion_matrix) {
                const ctxCM = document.getElementById('confusionMatrixChart').getContext('2d');
                new Chart(ctxCM, {
                    type: 'bar',
                    data: {
                        labels: ['True Real', 'False Fake', 'False Real', 'True Fake'],
                        datasets: [{
                            label: 'Count',
                            data: [
                                data.confusion_matrix.true_negative,
                                data.confusion_matrix.false_positive,
                                data.confusion_matrix.false_negative,
                                data.confusion_matrix.true_positive
                            ],
                            backgroundColor: [
                                'rgba(34, 197, 94, 0.7)', // True Negative (Real)
                                'rgba(239, 68, 68, 0.7)', // False Positive
                                'rgba(245, 158, 11, 0.7)', // False Negative
                                'rgba(34, 197, 94, 0.7)'  // True Positive (Fake)
                            ],
                            borderWidth: 1,
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { 
                                beginAtZero: true,
                                ticks: { color: '#94A3B8' },
                                grid: { color: 'rgba(255,255,255,0.05)' }
                            },
                            x: {
                                ticks: { color: '#94A3B8' },
                                grid: { display: false }
                            }
                        },
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }

            // Training Loss Chart
            if(document.getElementById('trainingLossChart') && data.training_history) {
                const ctxTL = document.getElementById('trainingLossChart').getContext('2d');
                new Chart(ctxTL, {
                    type: 'line',
                    data: {
                        labels: data.training_history.epochs,
                        datasets: [
                            {
                                label: 'Training Loss',
                                data: data.training_history.training_loss,
                                borderColor: 'rgba(59, 130, 246, 1)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                tension: 0.4,
                                fill: true
                            },
                            {
                                label: 'Validation Loss',
                                data: data.training_history.validation_loss,
                                borderColor: 'rgba(239, 68, 68, 1)',
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                tension: 0.4,
                                fill: true
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { 
                                beginAtZero: true,
                                ticks: { color: '#94A3B8' },
                                grid: { color: 'rgba(255,255,255,0.05)' }
                            },
                            x: {
                                ticks: { color: '#94A3B8' },
                                grid: { display: false }
                            }
                        },
                        plugins: {
                            legend: { position: 'bottom', labels: { color: '#F8FAFC' } }
                        }
                    }
                });
            }
        });
}

document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    animateCounters();
    initCharts();
    
    // Auto-fill progress bars on load (Home page)
    const progressBars = document.querySelectorAll('.progress-bar[data-width]');
    setTimeout(() => {
        progressBars.forEach(bar => {
            bar.style.width = bar.getAttribute('data-width');
        });
    }, 500);
});
