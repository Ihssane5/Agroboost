document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const input_temp = document.getElementById('temperature').value;
    const input_hum = document.getElementById('humidity').value;
    const input_mois = document.getElementById('moisture').value;
    const input_soltype = document.querySelector('input[name="soltype"]:checked').value;
    const input_croptype = document.querySelector('input[name="croptype"]:checked').value;
    const input_Nitro = document.getElementById('nitrogene').value;
    const input_Pota = document.getElementById('potassium').value;
    const input_Phos = document.getElementById('phosphorous').value;
    const input_data = {
        'temperature' : [input_temp],
        'humidity' : [input_hum],
        'moisture' : [input_mois],
        'soil type' : [input_soltype],
        'crop type' : [input_croptype],
        'nitrogen' : [input_Nitro],
        'potassium' : [input_Pota],
       'phosphorous' : [input_Phos]
    };
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({input_data})
    });
     result = await response.json();
     
    document.getElementById('fertilizer').innerText = `Top Fertilizer : ${result.fertilizer}`;
});

