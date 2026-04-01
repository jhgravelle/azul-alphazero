// frontend/game.js

const API = "http://127.0.0.1:8000";

async function loadState() {
  const response = await fetch(`${API}/state`);
  const state = await response.json();
  document.getElementById("debug").textContent = JSON.stringify(state, null, 2);
}

loadState();