document.addEventListener('DOMContentLoaded', () => {
    // ── Navigation Logic ──
    const navLinks = document.querySelectorAll('.nav-links li');
    const views = document.querySelectorAll('.view');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            // Update active nav
            navLinks.forEach(n => n.classList.remove('active'));
            link.classList.add('active');

            // Update active view
            const targetView = link.getAttribute('data-view');
            views.forEach(v => {
                v.classList.remove('active');
                if (v.id === `${targetView}-view`) {
                    v.classList.add('active');
                }
            });

            if (targetView === 'sandbox') {
                loadSandbox();
            }
        });
    });

    // ── API Fetchers ──
    async function fetchLeaderboard() {
        try {
            const res = await fetch('/api/leaderboard');
            if (!res.ok) throw new Error('Failed to fetch leaderboard');
            const data = await res.json();
            renderLeaderboard(data.leaderboard);
        } catch (err) {
            document.getElementById('leaderboard-body').innerHTML = `
                <tr><td colspan="10" class="loading-cell text-red">Error loading models: ${err.message}</td></tr>
            `;
        }
    }

    async function fetchHistoricalPrice(ticker, date) {
        try {
            const res = await fetch(`/api/historical-price?ticker=${ticker}&date=${date}`);
            if (!res.ok) return null;
            const data = await res.json();
            return data.price;
        } catch {
            return null;
        }
    }

    // ── Leaderboard Rendering ──
    function formatPct(val) {
        if (val === null || val === undefined) return '-';
        const sign = val > 0 ? '+' : '';
        const color = val > 0 ? 'text-green' : (val < 0 ? 'text-red' : '');
        return `<span class="${color}">${sign}${val.toFixed(2)}%</span>`;
    }

    function renderLeaderboard(stocks) {
        const tbody = document.getElementById('leaderboard-body');
        tbody.innerHTML = '';

        stocks.slice(0, 10).forEach((stock, index) => {
            // Main Row
            const tr = document.createElement('tr');
            tr.className = 'main-row';
            
            const probPct = (stock.doubling_prob_12m * 100).toFixed(1);
            const probColor = stock.doubling_prob_12m >= 0.5 ? 'var(--color-green)' : 'var(--text-muted)';
            
            // Calculate 52W slider position
            let sliderHtml = '<span class="text-muted">N/A</span>';
            if (stock.performance.low_52w && stock.performance.high_52w) {
                const low = stock.performance.low_52w;
                const high = stock.performance.high_52w;
                const cur = stock.current_price;
                const range = high - low;
                // clamp between 0 and 100
                const percent = range > 0 ? Math.max(0, Math.min(100, ((cur - low) / range) * 100)) : 50;
                
                sliderHtml = `
                    <div class="range-wrapper">
                        <span class="range-val">${low.toFixed(1)}</span>
                        <div class="range-bar-bg">
                            <div class="range-indicator" style="left: ${percent}%"></div>
                        </div>
                        <span class="range-val">${high.toFixed(1)}</span>
                    </div>
                `;
            }

            const pe = stock.features.pe_ratio !== null ? stock.features.pe_ratio.toFixed(1) : '-';

            tr.innerHTML = `
                <td>#${index + 1}</td>
                <td class="ticker-col">
                    <span class="ticker-sym">${stock.ticker}</span>
                    <span class="company-name">${stock.name}</span>
                </td>
                <td><span class="sector-badge">${stock.sector}</span></td>
                <td class="numeric">$${stock.current_price.toFixed(2)}</td>
                <td class="numeric">$${stock.market_cap_b.toFixed(1)}B</td>
                <td class="numeric">${pe}</td>
                <td>${sliderHtml}</td>
                <td class="numeric">${formatPct(stock.performance.ytd)}</td>
                <td class="numeric">${formatPct(stock.performance['1y'])}</td>
                <td class="numeric">${formatPct(stock.performance['2y'])}</td>
                <td class="numeric">${formatPct(stock.performance['3y'])}</td>
                <td class="prob-cell">
                    <div class="prob-val">${probPct}%</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: ${probPct}%; background: ${probColor}"></div>
                    </div>
                    <button class="btn-action add-sandbox-btn" data-ticker="${stock.ticker}" data-name="${stock.name}" data-price="${stock.current_price}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                    </button>
                </td>
            `;

            // Expanded Row (Analysis)
            const expandTr = document.createElement('tr');
            expandTr.className = 'analysis-row hidden';
            
            // Build SHAP DOM
            let shapHtml = '';
            stock.top_drivers.slice(0, 5).forEach(d => {
                const maxShap = Math.max(...stock.top_drivers.map(x => Math.abs(x.shap)));
                const width = (Math.abs(d.shap) / maxShap) * 100;
                const signClass = d.shap > 0 ? 'pos' : 'neg';
                
                shapHtml += `
                    <div class="shap-item">
                        <div class="shap-item-label">${d.feature.replace(/_/g, ' ')}</div>
                        <div class="shap-item-bar-container">
                            <div class="shap-item-bar ${signClass}" style="width: ${width}%"></div>
                        </div>
                    </div>
                `;
            });

            expandTr.innerHTML = `
                <td colspan="10">
                    <div class="analysis-content">
                        <div class="nlp-reasoning">
                            ${stock.nlp_reasoning}
                        </div>
                        <div class="shap-breakdown">
                            <div class="shap-column">
                                <h4>Top Model Drivers (SHAP)</h4>
                                ${shapHtml}
                            </div>
                        </div>
                    </div>
                </td>
            `;

            // Toggle expansion (ignore if clicking sandbox button)
            tr.addEventListener('click', (e) => {
                if (e.target.closest('.add-sandbox-btn')) return;
                expandTr.classList.toggle('hidden');
            });

            // Bind sandbox button
            const addBtn = tr.querySelector('.add-sandbox-btn');
            addBtn.addEventListener('click', () => {
                openSandboxModal(stock.ticker, stock.name, stock.current_price);
            });

            tbody.appendChild(tr);
            tbody.appendChild(expandTr);
        });
    }

    // ── Sandbox Modal Logic ──
    const modal = document.getElementById('sandbox-modal');
    const closeBtns = document.querySelectorAll('.close-modal, .close-modal-btn');
    const dateInput = document.getElementById('sandbox-date');
    const priceInput = document.getElementById('sandbox-buy-price');
    const errorText = document.getElementById('sandbox-price-error');
    const submitBtn = document.getElementById('sandbox-submit-btn');
    const form = document.getElementById('sandbox-form');

    function openSandboxModal(ticker, name, currentPrice) {
        document.getElementById('modal-ticker-title').innerText = `Add ${ticker} to Sandbox`;
        document.getElementById('modal-ticker-name').innerText = name;
        document.getElementById('sandbox-ticker').value = ticker;
        document.getElementById('sandbox-current-price').value = currentPrice;
        
        dateInput.value = '';
        priceInput.value = '';
        document.getElementById('sandbox-qty').value = '';
        errorText.classList.add('hidden');
        submitBtn.disabled = true;
        
        modal.classList.remove('hidden');
    }

    closeBtns.forEach(btn => btn.addEventListener('click', () => modal.classList.add('hidden')));

    // Date change -> fetch historical price
    dateInput.addEventListener('change', async (e) => {
        const date = e.target.value;
        const ticker = document.getElementById('sandbox-ticker').value;
        if (!date || !ticker) return;

        priceInput.placeholder = 'Fetching...';
        priceInput.value = '';
        submitBtn.disabled = true;
        errorText.classList.add('hidden');

        const price = await fetchHistoricalPrice(ticker, date);
        if (price !== null) {
            priceInput.value = price;
            submitBtn.disabled = false;
        } else {
            errorText.classList.remove('hidden');
            priceInput.placeholder = '';
        }
    });

    // Form submit
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const pwd = prompt("Enter Admin Password to update sandbox:");
        if (pwd === null) return; // User cancelled

        const position = {
            ticker: document.getElementById('sandbox-ticker').value,
            buy_date: dateInput.value,
            buy_price: parseFloat(priceInput.value),
            quantity: parseInt(document.getElementById('sandbox-qty').value, 10),
            current_price: parseFloat(document.getElementById('sandbox-current-price').value)
        };

        try {
            const res = await fetch('/api/sandbox/add', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Admin-Password': pwd
                },
                body: JSON.stringify(position)
            });
            
            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || 'Failed to add to sandbox');
            }
            
            modal.classList.add('hidden');
            
            // Switch to sandbox view to see it (index 2 because Home is 0, Leaderboard is 1, Sandbox is 2)
            navLinks[2].click();
        } catch (err) {
            alert('Error: ' + err.message);
        }
    });

    // ── Sandbox Rendering ──
    async function loadSandbox() {
        const tbody = document.getElementById('sandbox-body');
        tbody.innerHTML = '<tr><td colspan="7" class="loading-cell">Loading...</td></tr>';

        try {
            const res = await fetch('/api/sandbox');
            const data = await res.json();
            
            let totalInvested = 0;
            let currentValue = 0;
            
            tbody.innerHTML = '';
            
            if (data.positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="loading-cell">Sandbox is empty. Add stocks from the Leaderboard.</td></tr>';
            }

            data.positions.forEach(pos => {
                const invested = pos.buy_price * pos.quantity;
                const current = pos.current_price * pos.quantity;
                const pnl = current - invested;
                const pnlPct = (current / invested - 1) * 100;
                
                totalInvested += invested;
                currentValue += current;
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="ticker-sym">${pos.ticker}</td>
                    <td>${pos.buy_date}</td>
                    <td class="numeric">${pos.quantity}</td>
                    <td class="numeric">$${pos.buy_price.toFixed(2)}</td>
                    <td class="numeric">$${pos.current_price.toFixed(2)}</td>
                    <td class="numeric">${formatPct(pnlPct)}</td>
                    <td class="numeric">$${current.toFixed(2)}</td>
                `;
                tbody.appendChild(tr);
            });

            // Update stats
            document.getElementById('sandbox-total-invested').innerText = `$${totalInvested.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            document.getElementById('sandbox-current-value').innerText = `$${currentValue.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            
            const totalPnl = currentValue - totalInvested;
            const pnlEl = document.getElementById('sandbox-pnl');
            pnlEl.innerText = `$${Math.abs(totalPnl).toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            pnlEl.className = 'stat-value ' + (totalPnl >= 0 ? 'text-green' : 'text-red');
            if(totalPnl < 0) pnlEl.innerText = '-' + pnlEl.innerText;
            if(totalPnl > 0) pnlEl.innerText = '+' + pnlEl.innerText;

        } catch (err) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading-cell text-red">Failed to load sandbox data.</td></tr>';
        }
    }

    // Init
    fetchLeaderboard();
});
