const BACKEND_URL = "http://127.0.0.1:8000";

export async function runAortix(csvFile) {
  const formData = new FormData();
  formData.append("file", csvFile);

  const response = await fetch(`${BACKEND_URL}/run-aortix`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("Backend processing failed");
  }

  return await response.json();
}
