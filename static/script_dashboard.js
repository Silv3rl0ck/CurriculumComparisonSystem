document.addEventListener('DOMContentLoaded', () => {
  // initialize counters if missing
  if (!localStorage.getItem('totalComparisons')) {
    localStorage.setItem('totalComparisons', '0');
    localStorage.setItem('textSimCount', '0');
    localStorage.setItem('semanticSimCount', '0');
  }

  const total   = parseInt(localStorage.getItem('totalComparisons'), 10);
  const textSim = parseInt(localStorage.getItem('textSimCount'), 10);
  const semSim  = parseInt(localStorage.getItem('semanticSimCount'), 10);

  // update stat cards
  document.getElementById('totalComparisons').textContent   = total;
  document.getElementById('textSimCount').textContent       = textSim;
  document.getElementById('semanticSimCount').textContent   = semSim;

  // render only the donut chart
  createAlgorithmChart(textSim, semSim);
});

function createAlgorithmChart(textSim, semSim) {
  const ctx = document.getElementById('algorithmChart').getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Text Similarity', 'Semantic Similarity'],
      datasets: [{
        data: [textSim, semSim],
        backgroundColor: ['rgba(52,152,219,0.7)', 'rgba(46,204,113,0.7)'],
        borderColor: ['rgba(52,152,219,1)', 'rgba(46,204,113,1)'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { callbacks: {
          label: ctx => `${ctx.label}: ${ctx.parsed} (${(ctx.parsed / (textSim + semSim) * 100).toFixed(1)}%)`
        }}
      }
    }
  });
}
