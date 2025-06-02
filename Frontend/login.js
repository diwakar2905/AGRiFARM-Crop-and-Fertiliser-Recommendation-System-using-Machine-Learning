function login() {
  const username = document.getElementById("username").value;
  const password = document.getElementById("password").value;

  // Simple static login check (for demo purposes)
  if (username === "admin" && password === "12345") {
    window.location.href = "index.html";  // redirect to main page
  } else {
    document.getElementById("error-message").innerHTML = "Invalid username or password!";
  }
}