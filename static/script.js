document.addEventListener('DOMContentLoaded', () => {
    atualizarDashboard();

    document.getElementById('unidade-select').addEventListener('change', () => {
        atualizarDashboard();
    });
});

function atualizarDashboard() {
    const unidade = document.getElementById('unidade-select').value;
    const pais = document.getElementById('pais-select').value;

    let urlDados = `/dados?unidade=${encodeURIComponent(unidade)}`;
    if (pais) urlDados += `&pais=${encodeURIComponent(pais)}`;

    fetch(`/indicadores`)
        .then(res => res.json())
        .then(data => {
            document.getElementById('total').innerText = `Registros: ${data.total_registros}`;
            document.getElementById('media').innerText = `Média: ${data.media_valores}`;
            document.getElementById('max').innerText = `Máx: ${data.max_valor}`;
            document.getElementById('min').innerText = `Mín: ${data.min_valor}`;
        });

    fetch(urlDados)
        .then(res => res.json())
        .then(data => {
            const trace = {
                x: data.map(d => d.City),
                y: data.map(d => d.Value),
                type: 'bar',
                marker: { color: '#0074D9' }
            };
            const layout = {
                title: `Níveis por Cidade${pais ? ' em ' + pais : ''}`,
                xaxis: { title: 'Cidade' },
                yaxis: { title: 'Valor' }
            };
            Plotly.newPlot('grafico', [trace], layout);
        });
}

function popularPaises() {
    fetch('/paises')
        .then(res => res.json())
        .then(paises => {
            const select = document.getElementById('pais-select');
            paises.forEach(pais => {
                const option = document.createElement('option');
                option.value = pais;
                option.text = pais;
                select.appendChild(option);
            });
        });
}

document.addEventListener('DOMContentLoaded', () => {
    popularPaises();

    document.getElementById('unidade-select').addEventListener('change', atualizarDashboard);
    document.getElementById('pais-select').addEventListener('change', atualizarDashboard);

    atualizarDashboard();
});