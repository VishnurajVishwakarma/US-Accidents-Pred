// Initialize Map
const map = L.map('map', {
    fullscreenControl: true,
    zoomControl: false // Move to bottom right
}).setView([37.77, -122.41], 10);

L.control.zoom({ position: 'bottomright' }).addTo(map);

// OpenStreetMap HOT tiles
L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
}).addTo(map);

// Add Heatmap Layer
async function loadHeatmap() {
    try {
        let res = await fetch('/data/heatmap');
        let data = await res.json();
        let heatData = data.map(p => [p.Start_Lat, p.Start_Lng, p.Severity * 20]);
        L.heatLayer(heatData, {
            radius: 25, blur: 15, maxZoom: 14,
            gradient: {0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
        }).addTo(map);
    } catch(e) {}
}
loadHeatmap();

// Add Legend
var legend = L.control({position: 'bottomright'});
legend.onAdd = function (map) {
    var div = L.DomUtil.create('div', 'info legend');
    div.innerHTML += "<h4>Accident Risk Heatmap</h4>";
    div.innerHTML += "<i style='background:#0000ff'></i> Low Risk<br>";
    div.innerHTML += "<i style='background:#ffff00'></i> Medium Risk<br>";
    div.innerHTML += "<i style='background:#ff0000'></i> High Risk<br>";
    return div;
};
legend.addTo(map);

// Nominatim Custom Autocomplete Logic
let startCoord = null;
let endCoord = null;

function setupAutocomplete(inputId, suggId, isStart) {
    let input = document.getElementById(inputId);
    let suggBox = document.getElementById(suggId);
    let timeout = null;
    
    input.addEventListener('input', () => {
        clearTimeout(timeout);
        if (!input.value) { suggBox.style.display = 'none'; return; }
        
        timeout = setTimeout(async () => {
            let res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${input.value}&limit=5`);
            let data = await res.json();
            
            suggBox.innerHTML = '';
            suggBox.style.display = 'block';
            
            data.forEach(item => {
                let div = document.createElement('div');
                div.className = 'suggestion-item';
                div.innerText = item.display_name;
                div.onclick = () => {
                    input.value = item.display_name;
                    if (isStart) startCoord = L.latLng(item.lat, item.lon);
                    else endCoord = L.latLng(item.lat, item.lon);
                    
                    suggBox.style.display = 'none';
                    if (isStart) {
                        L.marker(startCoord).addTo(map).bindPopup("Source").openPopup();
                        map.panTo(startCoord);
                    } else {
                        L.marker(endCoord).addTo(map).bindPopup("Destination").openPopup();
                    }
                };
                suggBox.appendChild(div);
            });
        }, 400); // 400ms debounce
    });
    
    document.addEventListener('click', (e) => {
        if (e.target !== input) suggBox.style.display = 'none';
    });
}

setupAutocomplete('start-input', 'start-suggestions', true);
setupAutocomplete('end-input', 'end-suggestions', false);

// Leaflet Routing Machine Hidden Setup
let routingControl = L.Routing.control({
    waypoints: [],
    routeWhileDragging: false,
    showAlternatives: true,
    createMarker: function() { return null; }, // Hide native markers, we draw our own
    lineOptions: { styles: [{opacity: 0}] }, // Hide native lines
    altLineOptions: { styles: [{opacity: 0}] }
}).addTo(map);

let activeRouteLines = [];

window.findRoute = function() {
    if (!startCoord || !endCoord) return alert("Please type and select an address from the dropdown suggestions for both points!");
    
    document.getElementById('route-results').innerHTML = `
        <div class="empty-state">
            <h4 style="margin-top: 0; color: #1a73e8;">Processing Machine Learning Routes</h4>
            <div style="font-size: 13px; color: #555; text-align: left; padding: 10px; background: #f0f4f8; border-radius: 6px; margin-top: 10px;">
                <p style="margin: 0 0 5px 0;"><b>Status:</b> Initializing inference query...</p>
                <div style="font-size: 11px; margin-top: 8px;">
                    <i>Detailed Operations:</i>
                    <ul style="margin: 5px 0; padding-left: 15px; color: #444;">
                        <li>Calculating geometry permutations via OpenStreetMap routing servers.</li>
                        <li>Extracting coordinate batches and building predictive feature matrices.</li>
                        <li>Awaiting response from Flask backend predict() API.</li>
                        <li style="color: #cf1322;"><b>Note:</b> Serverless cold starts (Vercel/Render) may take 10-15 seconds to load the 280MB model into memory. Please wait...</li>
                    </ul>
                </div>
            </div>
        </div>`;
        
    // Clear previous custom lines
    activeRouteLines.forEach(l => map.removeLayer(l));
    activeRouteLines = [];
    
    routingControl.setWaypoints([startCoord, endCoord]);
}

routingControl.on('routesfound', async function(e) {
    let routes = e.routes;
    let resultsDiv = document.getElementById('route-results');
    resultsDiv.innerHTML = '';
    
    let evaluatedRoutes = [];
    let errorsEncountered = [];
    
    // Evaluate all routes through ML API
    for (let i = 0; i < routes.length; i++) {
        let route = routes[i];
        let coords = route.coordinates;
        
        let sampled = [];
        let step = Math.max(1, Math.floor(coords.length / 20)); // downsample coordinate density
        for(let j=0; j<coords.length; j+=step) {
            sampled.push({ Start_Lat: coords[j].lat, Start_Lng: coords[j].lng });
        }
        
        try {
            let req = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(sampled) });
            
            if (!req.ok) {
                let errorText = await req.text();
                throw new Error(`[Status ${req.status}] ${errorText}`);
            }
            
            let result = await req.json();
            
            if (!result.severities || result.severities.length === 0) {
                 throw new Error(`Invalid response format from /predict`);
            }
            
            let avgSeverity = result.severities.reduce((a, b) => a + b, 0) / result.severities.length;
            let penaltyWeight = 10000;
            let customCost = route.summary.totalDistance + (avgSeverity * penaltyWeight);
            
            evaluatedRoutes.push({
                index: i,
                route: route,
                avgSeverity: avgSeverity,
                customCost: customCost,
                distance: (route.summary.totalDistance / 1609.34).toFixed(1) + ' mi',
                time: Math.round(route.summary.totalTime / 60) + ' min'
            });
        } catch (err) {
            console.error(`Route ${i+1} Evaluation Error:`, err);
            errorsEncountered.push(`Route ${i+1} Failed: ${err.message}`);
        }
    }
    
    if (evaluatedRoutes.length === 0) {
        resultsDiv.innerHTML = `
            <div class="empty-state" style="border: 1px solid #ff4d4f; background-color: #fff1f0; padding: 15px; border-radius: 8px;">
                <h4 style="color: #cf1322; margin-top: 0;">Server Error Processing Routes</h4>
                <p style="font-size: 13px; color: #5c0011;">The ML backend failed to evaluate the paths.</p>
                <div style="font-size: 11px; background: rgba(0,0,0,0.05); padding: 5px; text-align: left; overflow-x: auto; white-space: pre-wrap;">
                    ${errorsEncountered.join('<br>')}
                </div>
                <p style="font-size: 12px; margin-top:10px;"><b>Common Vercel issues:</b> 504 Gateway Timeout (model took too long to load), Serverless size limit exceeded, or App logic crash. Check Vercel Logs for root cause.</p>
            </div>
        `;
        return;
    }
    
    // Find shortest route by distance
    let shortestRouteIndex = -1;
    let minDistance = Infinity;
    evaluatedRoutes.forEach((er) => {
        if (er.route.summary.totalDistance < minDistance) {
            minDistance = er.route.summary.totalDistance;
            shortestRouteIndex = er.index;
        }
    });
    
    // Sort by safest (minimum custom cost)
    evaluatedRoutes.sort((a,b) => a.customCost - b.customCost);
    
    evaluatedRoutes.forEach((er, idx) => {
        let isSafest = idx === 0; // The first one after sort is the absolute safest
        let isShortest = er.index === shortestRouteIndex;
        
        let routeTitle = "Alternative Route";
        let badgeClass = 'badge-moderate';
        let badgeText = 'Moderate Risk';

        if (isSafest && isShortest) {
            routeTitle = "Shortest & Safest Route";
            badgeClass = 'badge-safe';
            badgeText = 'ML Verified';
        } else if (isSafest) {
            routeTitle = "Safest Route (Alternative)";
            badgeClass = 'badge-safe';
            badgeText = 'ML Recommended';
        } else if (isShortest) {
            routeTitle = "Shortest Route";
            badgeClass = er.avgSeverity > 2.5 ? 'badge-danger' : 'badge-moderate';
            badgeText = er.avgSeverity > 2.5 ? 'High Risk Danger' : 'Moderate Risk';
        } else {
            routeTitle = "Alternative Route";
            badgeClass = er.avgSeverity > 2.5 ? 'badge-danger' : 'badge-moderate';
            badgeText = er.avgSeverity > 2.5 ? 'High Risk' : 'Moderate Risk';
        }
        
        let card = document.createElement('div');
        card.className = `route-card ${isSafest ? 'selected-safe' : ''}`;
        card.innerHTML = `
            <div class="route-title">
                ${routeTitle}
                <span class="badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="route-meta">
                <strong>${er.time}</strong> (${er.distance})
            </div>
            <div style="font-size:12px; color:#5f6368;">Avg severity score: ${er.avgSeverity.toFixed(2)}</div>
        `;
        
        // Custom draw line
        let color = isSafest ? '#1a73e8' : '#9aa0a6';
        let weight = isSafest ? 8 : 5;
        let polyline = L.polyline(er.route.coordinates, { color, weight, opacity: 0.8 }).addTo(map);
        activeRouteLines.push(polyline);
        
        if (isSafest) polyline.bringToFront();
        
        card.onclick = () => {
            // reset UI highlights
            document.querySelectorAll('.route-card').forEach(c => c.className = 'route-card');
            card.classList.add('selected');
            
            // visually highlight selected route on map
            activeRouteLines.forEach(l => {
               l.setStyle({color: '#9aa0a6', weight: 5}); 
               l.bringToBack();
            });
            polyline.setStyle({color: '#1a73e8', weight: 8});
            polyline.bringToFront();
        }
        
        resultsDiv.appendChild(card);
    });
});
