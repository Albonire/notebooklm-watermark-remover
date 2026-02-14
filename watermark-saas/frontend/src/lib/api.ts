const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

interface UploadResponse {
  jobId: string;
  uploadUrl: string;
  expiresIn: number;
  remainingCredits: number;
}

interface JobStatusResponse {
  jobId: string;
  status: "pending" | "processing" | "completed" | "failed";
  filename?: string;
  createdAt?: number;
  previewUrl?: string;
  error?: string;
}

interface DownloadResponse {
  downloadUrl: string;
  expiresIn: number;
  filename?: string;
}

export async function requestUpload(
  filename: string,
  contentType: string
): Promise<UploadResponse> {
  const res = await fetch(`${API_URL}/upload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, contentType }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Upload request failed (${res.status})`);
  }

  return res.json();
}

export async function uploadFile(
  presignedUrl: string,
  file: File
): Promise<void> {
  const res = await fetch(presignedUrl, {
    method: "PUT",
    headers: { "Content-Type": file.type },
    body: file,
  });

  if (!res.ok) {
    throw new Error(`File upload failed (${res.status})`);
  }
}

export async function getJobStatus(
  jobId: string
): Promise<JobStatusResponse> {
  const res = await fetch(`${API_URL}/jobs/${jobId}/status`);

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Status check failed (${res.status})`);
  }

  return res.json();
}

export async function requestDownload(
  jobId: string
): Promise<DownloadResponse> {
  const res = await fetch(`${API_URL}/jobs/${jobId}/download`, {
    method: "POST",
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Download request failed (${res.status})`);
  }

  return res.json();
}
