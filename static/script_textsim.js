document.addEventListener('DOMContentLoaded', function() {
    const compareBtn = document.getElementById('compareBtn');
    const input1 = document.getElementById('input1');
    const input2 = document.getElementById('input2');
    const algorithmSelect = document.getElementById('algorithm');
    const similarityScore = document.getElementById('similarityScore');

    compareBtn.addEventListener('click', function() {
        const text1 = input1.value.trim();
        const text2 = input2.value.trim();
        const algorithm = algorithmSelect.value;
        
        if (!text1 || !text2) {
            alert('Please enter text in both boxes.');
            return;
        }

        let score;
        switch (algorithm) {
            case 'jaccard':
                score = calculateJaccardSimilarity(text1, text2);
                break;
            case 'cosine':
                score = calculateCosineSimilarity(text1, text2);
                break;
            case 'levenshtein':
                score = calculateLevenshteinSimilarity(text1, text2);
                break;
            default:
                score = calculateLevenshteinSimilarity(text1, text2);
        }

        // display result
        similarityScore.textContent = `${Math.round(score * 100)}%`;

        // ── update stats ────────────────────────────────────────────────
        const total    = parseInt(localStorage.getItem('totalComparisons') || '0', 10) + 1;
        const textSim  = parseInt(localStorage.getItem('textSimCount')   || '0', 10) + 1;
        localStorage.setItem('totalComparisons', total);
        localStorage.setItem('textSimCount', textSim);
    });

    // Jaccard Similarity
    function calculateJaccardSimilarity(text1, text2) {
        const tokenize = text => new Set(
            text.toLowerCase()
                .replace(/[^\w\s]/g, '')
                .split(/\s+/)
                .filter(w => w.length)
        );
        const set1 = tokenize(text1), set2 = tokenize(text2);
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        return union.size ? intersection.size / union.size : 0;
    }

    // Cosine Similarity
    function calculateCosineSimilarity(text1, text2) {
        const tokenize = text => text.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/).filter(w => w);
        const tokens1 = tokenize(text1), tokens2 = tokenize(text2);
        const all = [...new Set([...tokens1, ...tokens2])];
        const v1 = all.map(t => tokens1.filter(x => x === t).length);
        const v2 = all.map(t => tokens2.filter(x => x === t).length);
        const dot = v1.reduce((s, n, i) => s + n * v2[i], 0);
        const mag1 = Math.sqrt(v1.reduce((s, n) => s + n*n, 0));
        const mag2 = Math.sqrt(v2.reduce((s, n) => s + n*n, 0));
        return (mag1 && mag2) ? dot / (mag1 * mag2) : 0;
    }

    // Levenshtein Similarity
    function calculateLevenshteinSimilarity(text1, text2) {
        // Compute the Levenshtein distance between text1 and text2
        function levenshteinDistance(s, t) {
            const m = s.length, n = t.length;
            if (m === 0) return n;
            if (n === 0) return m;

            // initialize dp table
            const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
            for (let i = 0; i <= m; i++) dp[i][0] = i;
            for (let j = 0; j <= n; j++) dp[0][j] = j;

            for (let i = 1; i <= m; i++) {
                for (let j = 1; j <= n; j++) {
                    const cost = s[i - 1] === t[j - 1] ? 0 : 1;
                    dp[i][j] = Math.min(
                        dp[i - 1][j] + 1,      // deletion
                        dp[i][j - 1] + 1,      // insertion
                        dp[i - 1][j - 1] + cost // substitution
                    );
                }
            }
            return dp[m][n];
        }

        const distance = levenshteinDistance(text1, text2);
        const maxLen = Math.max(text1.length, text2.length);
        return maxLen === 0 ? 1 : 1 - distance / maxLen;
    }
});

