function predictCrop() {
  const N = document.getElementById("N").value.trim();
  const P = document.getElementById("P").value.trim();
  const K = document.getElementById("K").value.trim();
  const temperature = document.getElementById("temperature").value.trim();
  const humidity = document.getElementById("humidity").value.trim();
  const ph = document.getElementById("ph").value.trim();
  const rainfall = document.getElementById("rainfall").value.trim();

  // Basic validation
  if (!N || !P || !K || !temperature || !humidity || !ph || !rainfall) {
    showResult("Please fill in all fields.", true);
    return;
  }

  const data = {
    N: parseFloat(N),
    P: parseFloat(P),
    K: parseFloat(K),
    temperature: parseFloat(temperature),
    humidity: parseFloat(humidity),
    ph: parseFloat(ph),
    rainfall: parseFloat(rainfall)
  };

  showResult("Predicting...", false);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  })
    .then(response => {
      if (!response.ok) throw new Error("Server error");
      return response.json();
    })
    .then(result => {
      if (result.recommended_crop) {
        showResult("🌱 <b>Recommended Crop:</b> " + result.recommended_crop, false);
      } else if (result.error) {
        showResult("Error: " + result.error, true);
      } else {
        showResult("Unexpected response from server.", true);
      }
    })
    .catch(error => {
      showResult("Prediction failed. Please try again later.", true);
      console.error("Error:", error);
    });
}

function predictFertilizer() {
  const N = document.getElementById("N").value.trim();
  const P = document.getElementById("P").value.trim();
  const K = document.getElementById("K").value.trim();
  const temperature = document.getElementById("temperature").value.trim();
  const humidity = document.getElementById("humidity").value.trim();
  const ph = document.getElementById("ph").value.trim();
  const rainfall = document.getElementById("rainfall").value.trim();

  // Basic validation
  if (!N || !P || !K || !temperature || !humidity || !ph || !rainfall) {
    showResult("Please fill in all fields.", true);
    return;
  }

  const data = {
    N: parseFloat(N),
    P: parseFloat(P),
    K: parseFloat(K),
    temperature: parseFloat(temperature),
    humidity: parseFloat(humidity),
    ph: parseFloat(ph),
    rainfall: parseFloat(rainfall)
  };

  showResult("Predicting...", false);

  fetch("http://127.0.0.1:5000/predict_fertilizer", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  })
    .then(response => {
      if (!response.ok) throw new Error("Server error");
      return response.json();
    })
    .then(result => {
      if (result.recommended_fertilizer) {
        showResult("🌱 <b>Recommended Fertilizer:</b> " + result.recommended_fertilizer, false);
      } else if (result.error) {
        showResult("Error: " + result.error, true);
      } else {
        showResult("Unexpected response from server.", true);
      }
    })
    .catch(error => {
      showResult("Prediction failed. Please try again later.", true);
      console.error("Error:", error);
    });
}

function showResult(message, isError) {
  const result = document.getElementById("result");
  result.innerHTML = message;
  result.style.color = isError ? "#c0392b" : "#2a4d69";
}